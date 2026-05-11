import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os

class RLLogger:
    def __init__(self, log_dir = "logs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def log_scalar(self, tag, value, step):
        """同时记录到 TensorBoard 并打印"""
        self.writer.add_scalar(tag, value, step)

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        self.writer.close()