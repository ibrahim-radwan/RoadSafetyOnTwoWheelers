# Shared utility functions

import logging
from config_params import CFGS


def setup_logger(name, level=CFGS.LOG_LEVEL):
    """Set up a logger with consistent formatting across all modules
    
    Args:
        name: Name of the logger (typically the module or class name)
        level: Logging level (from CFGS.LOG_LEVEL by default)
    
    Returns:
        logger: Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"%(asctime)s - {name} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger
