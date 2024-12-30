import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler

class trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.learn_rate = config.learn_rate
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.num_epochs = config.num_epochs
        self.max_norm = config.max_norm
        
        #Set device based on training configuration
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device:", self.device)

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            sampler=RandomSampler(self.dataset, replacement=True),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(self.num_epochs):
            print("epoch:", epoch)

            #Iterate through each training batch
            for (inputs, targets) in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                optimizer.step()

            print("loss:", loss.item())