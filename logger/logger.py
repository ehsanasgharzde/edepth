import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(file: str | Path):
    # Extract the module name from the relative path and convert .py to .log
    file_name = file.split("edepth")[-1]
    log_file_name = file_name.split('.')[0] + '.log'
    
    # Extract directory path from relative_path (e.g., 'models' from 'models/my_model.py')
    dir_path = os.path.dirname(file_name)
    
    # Construct log directory path (e.g., 'logs/models')
    log_dir = os.path.join(Path(__file__).parent, 'logs', dir_path)
    
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path to the log file (e.g., 'logs/models/my_model.log')
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # Get logger instance with the module name
    logger = logging.getLogger(os.path.splitext(file_name)[0])
    logger.setLevel(logging.INFO)  # Default level is INFO
    
    # Check if handlers are already attached to avoid duplicates
    if not logger.handlers:
        # Create formatter with timestamp, name,  level name, and message
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create file handler with UTF-8 encoding and rotation
        file_handler = RotatingFileHandler(
            log_file_path, 
            maxBytes=10485760,  # 10 MB
            backupCount=5,      # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
