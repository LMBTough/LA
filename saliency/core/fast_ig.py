import torch.nn as nn
import torch.nn.functional as F
import torch
class FastIG:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, data, target):
        data.requires_grad_()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        return (data * data.grad).detach().cpu().numpy()
    
    
class FastIGKL:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, data, target):
        data.requires_grad_()
        output = self.model(data)
        loss = -F.kl_div(output.log(), torch.ones_like(output) / 1000, reduction="batchmean")
        loss.backward()
        return (data * data.grad).detach().cpu().numpy()