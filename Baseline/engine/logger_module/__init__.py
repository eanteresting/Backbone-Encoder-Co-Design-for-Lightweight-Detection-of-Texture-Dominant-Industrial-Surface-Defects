import sys, logging
import torch.distributed as dist

def get_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    is_main_process = True
    if dist.is_available() and dist.is_initialized():
        is_main_process = dist.get_rank() == 0

    if not logger.handlers and is_main_process:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
