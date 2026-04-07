"""Training runner: composition root for a full training loop."""                
                                                                                   
from __future__ import annotations
                                                                                
import random                                                                  
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
                                                                                
from robovla.config import SyntheticDataConfig, TrainingConfig, VLAPolicyConfig  
from robovla.data.sample import BatchedSample                                    
from robovla.data.synthetic import SyntheticDataset                              
from robovla.models.policy import VLAPolicy                                    
from robovla.utils.logging_utils import custom_logger                            
from robovla.utils.training_utils.collate import collate_fn                    
                                                                                
                                                                                
class Runner:
    """Orchestrates training of a VLA policy on synthetic data.
                                                                                
    A Runner owns its own configs, policy, optimizer, dataloader, logger,        
    and checkpoint directory. Call ``runner.train()`` to start the loop.         
    """                                                                                                                                  
    def __init__(self,policy_config: VLAPolicyConfig,data_config: SyntheticDataConfig,train_config: TrainingConfig) -> None:

        self.policy_config = policy_config
        self.data_config = data_config                                           
        self.train_config = train_config                                       
                                                                                
        # Seed - reproducibility
        torch.manual_seed(train_config.seed)                                     
        np.random.seed(train_config.seed)                                        
        random.seed(train_config.seed)
                                                                                
        # Device - mps, cpu, gpu                                                 
        self.device = torch.device(train_config.device)

        # Policy
        self.policy = VLAPolicy(policy_config).to(self.device)                   
                                                                                
        # Dataset + dataloader                                                
        self.dataset = SyntheticDataset(
            data_config,                                                         
            preprocess=self.policy.vision.preprocess,                          
        )                                                                        
        self._collate = collate_fn(self.policy.text.tokenizer)

        self.loader = DataLoader(                                                
            self.dataset,                                                      
            batch_size=train_config.batch_size,                                  
            shuffle=True,                                                        
            collate_fn=self._collate,
            drop_last=True,                                                      
        )                                                                      

        # Optimizer: only trainable parameters (frozen CLIP stays frozen)     
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]      

        # Weighted Adam                                                           
        self.optimizer = torch.optim.AdamW(                                    
            trainable_params,                                                    
            lr=train_config.learning_rate,                                     
            weight_decay=train_config.weight_decay,
        )                                                                        

        # Materialize output dirs                                             
        self.artifacts_dir: Path = train_config.artifacts_dir                  
        self.logs_dir: Path = train_config.logs_dir                              
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)                         
                                                                                
        # Logger                                                              
        self.logger = custom_logger(                                           
            name="robovla.training",                                             
            logs_dir=self.logs_dir,
            run_name=train_config.run_name,                                      
        )                                                                      

        # Log configs so the run is handled                   
        self.logger.info("Runner initialized")
        self.logger.info(f"device={self.device}")                                
        self.logger.info(f"policy_config={policy_config.model_dump_json()}")     
        self.logger.info(f"data_config={data_config.model_dump_json()}")         
        self.logger.info(f"train_config={train_config.model_dump_json()}")   

        # Sanity check to check percentage of trainable parameters.    
        n_trainable = sum(p.numel() for p in trainable_params)                 
        n_total = sum(p.numel() for p in self.policy.parameters())               
        self.logger.info(f"params: trainable={n_trainable:,} total={n_total:,}")
                                                                                
    def train(self) -> None:                                                     
        """Run the main training loop."""                                        
        self.policy.train()                                                      
        self.logger.info(f"Starting training for {self.train_config.num_steps} steps")                                                                          

        loader_iter = iter(self.loader)                                          
        for step in range(self.train_config.num_steps):                        
            try:
                batch = next(loader_iter)
            except StopIteration:                                                
                loader_iter = iter(self.loader)
                batch = next(loader_iter)                                        
                                                                                
            batch = self._move_batch_to_device(batch)                            

            loss = self.policy(                                                  
                images=batch.images,                                           
                text_ids=batch.text_ids,                                         
                proprio=batch.proprios,
                action_chunks=batch.action_chunks,                               
            )                                                                    

            self.optimizer.zero_grad(set_to_none=True)                           
            loss.backward()                                                    
            grad_norm = torch.nn.utils.clip_grad_norm_(                          
                self.policy.parameters(),                                      
                self.train_config.grad_clip,
            )                                                                    
            self.optimizer.step()
                                                                                
            if step % self.train_config.log_every == 0:                          
                self.logger.info(
                    f"step={step:05d} loss={loss.item():.4f} grad_norm={grad_norm.item():.3f}"                                                
                )
                                                                                
            if (step + 1) % self.train_config.checkpoint_every == 0:             
                self._save_checkpoint(step + 1)
                                                                                
        # Final checkpoint at the end of training                              
        self._save_checkpoint(self.train_config.num_steps)
        self.logger.info("Training complete")                                    

    def _move_batch_to_device(self, batch: BatchedSample) -> BatchedSample:   
        """Take tensors and move to compatible device"""   
        return BatchedSample(                                                  
            images=batch.images.to(self.device),                                 
            text_ids=batch.text_ids.to(self.device),
            proprios=batch.proprios.to(self.device),                             
            action_chunks=batch.action_chunks.to(self.device),                   
        )
                                                                                
    def _save_checkpoint(self, step: int) -> None:                             
        """Save policy, optimizer, and configs to a timestamped .pt file."""
        ckpt_path = self.artifacts_dir/f"{self.train_config.run_name}_step{step:06d}.pt"                                                                 
        
        # Don't like Any but I need to look up the types - its generic.
        payload: dict[str, Any] = {                                            
            "step": step,                                                        
            "policy_state_dict": self.policy.state_dict(),                     
            "optimizer_state_dict": self.optimizer.state_dict(),                 
            "policy_config": self.policy_config.model_dump(),
            "data_config": self.data_config.model_dump(),                        
            "train_config": self.train_config.model_dump(mode="json"),           
        }
        torch.save(payload, ckpt_path)                                           
        self.logger.info(f"Saved checkpoint to {ckpt_path}")