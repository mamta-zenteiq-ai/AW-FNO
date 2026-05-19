import torch

class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        # Compute mean and std across batch and all spatial dimensions (keep channels)
        dims = list(range(x.ndim))
        if len(dims) > 1:
            dims.pop(1) # Remove channel dimension from reduction
        self.mean = torch.mean(x, dim=dims, keepdim=True)
        self.std = torch.std(x, dim=dims, keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self