import time

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler

class trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.optimizer = config.optimizer
        
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available else "cpu"
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device:", self.device)

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            sampler=RandomSampler(self.dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=
        )