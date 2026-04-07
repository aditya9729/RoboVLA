# RoboVLA

A minimal, training-ready **Vision-Language-Action (VLA) policy** skeleton for
robotic manipulation, built around a **flow matching action head**. The focus
is on clean structure and shape-correct forward passes — not task performance.

---

## Architecture

```
  Image (B,3,H,W) ────> VisionEncoder (CLIP ViT-B/32, frozen) ──> (B, N_v, D)
                                                                       │
  Text  (str list) ───> TextEncoder  (CLIP text tower, frozen) ──> (B, N_t, D)
                                                                       │
  Proprio (B, P)  ────> ProprioEncoder (MLP)                  ──> (B, N_p, D)
                                                                       │
                                                                       ▼
                              FusionTransformer (self-attention)
                              + learned modality-type embeddings
                                                                       │
                                                                       ▼
                                 context tokens (B, N, D)
                                                                       │
                                                                       ▼
                       FlowMatchingActionHead (DiT-style cross-attention)
                        ├─ compute_loss(action_chunks, context) -> MSE on velocity
                        └─ sample(context)                      -> (B, H, A)
```

**Key design choices:**

- **Shared `embed_dim`** (default 512) across vision, text, proprio, fusion, and
the action head so that multimodal concatenation is trivially typed.
- **Frozen CLIP by default.** Only the projection heads, fusion transformer,
and flow matching denoiser are trained. This preserves internet-scale
pre-training and keeps trainable parameter count small.
- **Per-patch / per-token features** are extracted from CLIP (not the pooled
CLS/EOT vectors). The fusion transformer attends over a rich spatial +
linguistic context rather than a single pooled embedding per modality.
- **Action chunking.** The policy predicts `action_horizon` future steps per
forward pass (à la pi0 and Diffusion Policy).
- **Rectified flow matching.** Straight-line interpolation between data and
noise, constant target velocity `v = a_1 − a_0`, Euler-integrated at
sampling time in a small number of steps (default 8).
- **Modality-type embeddings.** A learned 3-way embedding tags each token
stream before concatenation so the fusion transformer can distinguish
vision / language / proprio tokens without additional positional encoding.

---

## Repository layout

```
RoboVLA/
├── pyproject.toml
├── README.md
├── artifacts/              # checkpoints (.pt) — gitignored
├── logs/                   # timestamped run logs — gitignored
├── src/robovla/
│   ├── config.py           # Pydantic configs: model / data / training
│   ├── cli.py              # `robovla train` and `robovla sample`
│   ├── data/
│   │   ├── sample.py       # Sample / BatchedSample NamedTuples
│   │   └── synthetic.py    # SyntheticDataset (random toy tensors)
│   ├── models/
│   │   ├── encoders.py     # VisionEncoder, TextEncoder, ProprioEncoder
│   │   ├── fusion.py       # FusionTransformer
│   │   ├── flow_matching.py  # FlowMatchingDenoiser + FlowMatchingActionHead
│   │   └── policy.py       # VLAPolicy (top-level)
│   ├── training/
│   │   └── runner.py       # Runner: training loop + checkpointing
│   ├── inference/
│   │   ├── checkpoint.py   # find_latest_checkpoint, load_policy_from_checkpoint
│   │   └── runner.py       # InferenceRunner with act() / act_batch()
│   └── utils/
│       ├── logging_utils/  # file + stdout logger factory
│       └── training_utils/ # custom collate function
└── tests/
    ├── test_policy_shapes.py
    └── test_checkpoint_roundtrip.py
```

---

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

First run of CLIP downloads ~~150 MB of weights into `~~/.cache/clip`.

---

## Usage

### Train on synthetic data

```bash
robovla train --run-name demo --num-steps 50 --device cpu
```

Checkpoints land in `artifacts/`, logs in `logs/`. Both are configurable via
CLI flags.

### Sample an action chunk from the latest checkpoint

```bash
robovla sample --instruction "pick up the red block"
```

This loads the most recent checkpoint, feeds a grey dummy image and a zero
proprio vector, and prints the sampled action chunk of shape
`(action_horizon, action_dim)`.

### Run the tests

```bash
pytest tests/ -v
```

Five tests covering:

- Forward-pass loss shape + finiteness
- `sample()` output shape
- Proprio shape-mismatch edge case
- Checkpoint discovery (`find_latest_checkpoint`)
- Round-trip equivalence of `encode_context` between trained and loaded policy

---

## Assumptions

1. **Data is synthetic.** Every tensor is a fresh random draw per
  `__getitem__`; nothing is persisted. Real teleop would swap
   `SyntheticDataset` for a `torch.utils.data.Dataset` backed by disk or a
   memory-mapped format.
2. **Frozen CLIP.** The vision and text towers are frozen by default. Only
  the projection layers, fusion, and action head contain trainable weights.
3. **Action space is continuous and normalized.** Flow matching assumes
  actions live in a scale compatible with `N(0, I)` noise. Real teleop data
   would need per-dimension normalization computed once over the training set.
4. **Single image, single timestep of proprio.** The architecture is ready for
  multi-frame / history inputs (the vision and proprio encoders already emit
   token sequences), but the current dataset and forward signature only
   consume one frame per sample.
5. **CPU-friendly defaults.** All configs default to dimensions and layer
  counts that fit in a laptop CPU run so the whole pipeline can be tested
   without a GPU.

---

## Extending to real robot teleop data

A rough checklist for turning this skeleton into a trainable real-data policy:

1. **Dataset.** Replace `SyntheticDataset` with a reader over your teleop
  format (e.g., LeRobot / RLDS / an HDF5 episode store). Emit the same
   `Sample` NamedTuple so the rest of the pipeline is untouched.
2. **Action normalization.** Compute per-dimension mean/std over the training
  set once and store them alongside the checkpoint. Normalize actions before
   the FM loss, denormalize after sampling.
3. **History + multi-view.** Feed `(B, T, 3, H, W)` through the vision
  encoder by folding `T` into the batch dim. Concatenate the resulting
   token sequences along the time dimension before fusion. Same trick works
   for multiple cameras.
4. **Proprio history.** `ProprioEncoder` already supports `num_tokens > 1`;
  stack a window of proprio frames along that axis.
5. **Unfreezing.** Once the action head converges, unfreeze the fusion
  transformer. Only unfreeze CLIP as a last resort and with a much smaller
   learning rate — otherwise you risk catastrophic forgetting of the
   internet-scale pretraining (same rationale as pi0's stop-gradient between
   VLM backbone and action expert).
6. **Validation loop.** Add a held-out validation split and log MSE + an
  action-space L2 metric every N steps.
7. **Exported inference.** Replace `torch.load(..., weights_only=False)` with
  a safer format: save weights via `torch.save(state_dict, ...)` and
   configs as JSON next to the checkpoint. This removes the pickle
   security surface for deployed policies.
8. **Serving.** `InferenceRunner.act()` is already the single-robot API.
  For deployment, wrap it in a small FastAPI or ROS 2 node that accepts
   `(image, instruction, proprio)` and returns an action chunk OR for more optimized runs - build a custom IR via ONNX format and use torchserve or use vllm.

---

## What was intentionally left out

- **Training to convergence** on real data.
- **Validation / early stopping** (no held-out data in a synthetic setting).
- **Mixed precision / gradient accumulation** (not needed for a toy run).
- **Hyperparameter search** (configs are defaults, not optimized).

---

## What I'd add next

- **Share one CLIP instance** between `VisionEncoder` and `TextEncoder` — currently loaded twice.
- **Multi-camera + history** — fold camera / time indices into the batch dim; fusion handles variable-length sequences natively.
- **Action + proprio normalization** computed once over the training set and stored in the checkpoint.
- **Key-padding masks** from the CLIP tokenizer so fusion doesn't attend over text padding.
- **AdaLN time conditioning** in the flow matching denoiser (real DiT pattern) instead of additive time embeddings.
- **Logit-normal FM time sampling** (SD3 trick) — one-line change, measurable gains.
- **BC → Offline RL → On-robot RL** (pi0 → pi*0 recipe). Start with advantage-weighted BC; it reuses the FM loss unchanged. Full on-robot RL only after BC+offline is solid.
- **Uncertainty-aware ensemble heads** to detect out-of-distribution states and trigger a safety fallback.
- **MPC safety filter** on top of sampled action chunks (reachability + collision check), with a stabilizing low-level fallback.
- **Hierarchical composition** — a high-level VLM/LLM emits subgoals, this policy executes them (pi0.5 pattern).
- **Test-time LoRA adaptation** for few-shot deployment to new robots / environments.
- **ONNX / TorchScript export** of the action head (freeze CLIP, precompute embeddings) for real-time control; TensorRT on-robot.
- **Determinism + shadow-mode deployment** for reproducible evals and zero-risk online logging alongside a teleoperator.

---

## Attribution

- **CLIP vision + text towers** via
`[open_clip_torch](https://github.com/mlfoundations/open_clip)`
(OpenAI CLIP weights).
- **Rectified flow matching** follows the formulation from
Lipman et al., *Flow Matching for Generative Modeling* (2023) and
Liu et al., *Rectified Flow* (2022).
- **Action chunking + cross-attention action head** inspired by
the **pi0** (Physical Intelligence, 2024) and **Diffusion Policy** (Chi et al., 2023)
architectures.

