import torch
import torch.nn as nn

class WavLinear(torch.nn.Module):
    # WavNet (architecture for wave form conversion piano to guitar)

    def __init__(self):
        super(WavLinear, self).__init__()

        # Basic Linear Transformer
        # resampled using sample rate = 5000
        self.l1 = nn.Linear(15000, 15000)

    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        x = self.l1(x)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))

        return x