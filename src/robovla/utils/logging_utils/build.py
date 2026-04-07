"""Logger factory that writes to both console and a timestamped log file."""
                                                                                   
from __future__ import annotations                                               
                                                                                
import logging                                                                   
import sys                                                                     
from datetime import datetime
from pathlib import Path


def custom_logger(name: str,logs_dir: Path, run_name: str,level: int = logging.INFO) -> logging.Logger:                                                           
    """Build a logger that writes to stdout and a timestamped log file.          

    The file is written to:                                                      
        logs_dir / f"{run_name}_{YYYYMMDD_HHMMSS}.log"                         
                                                                                
    Args:
        name: Logger name (used by ``logging.getLogger``).                       
        logs_dir: Directory to write the log file into. Created if missing.      
        run_name: Identifier used as filename prefix.                            
        level: Standard logging level (default INFO).                            
                                                                                
    Returns:                                                                     
        A configured ``logging.Logger`` ready to use.                          
    """                                                                          
    logs_dir.mkdir(parents=True, exist_ok=True)
                                                                                
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                         
    log_path = logs_dir / f"{run_name}_{timestamp}.log"
                                                                                
    logger = logging.getLogger(name)                                             
    logger.setLevel(level)

    logger.handlers.clear()                                                      
    # Prevent propagation to the root logger   
    logger.propagate = False                                                     
                                                                                
    formatter = logging.Formatter(                                               
        fmt="%(asctime)s | %(levelname)s | %(message)s",                       
        datefmt="%Y-%m-%d %H:%M:%S",                                             
    )
                                                                                
    file_handler = logging.FileHandler(log_path)                                 
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)                                              
                                                                                
    stream_handler = logging.StreamHandler(sys.stdout)                           
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)                                            
                                                                                
    return logger