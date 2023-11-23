import torch
import torch.nn as nn


class L2E(nn.Module):
    def __init__(self, nl=2, nz=64):
        super().__init__()
        self.nl = nl
        self.nz = nz

        self.main = nn.Sequential(
            nn.Linear(nl, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),

            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),

            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),

            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),

            nn.Linear(nz, nz),
        )

    def forward(self, y):
        y = y.view(-1, self.nl) + 1e-8
        y = self.main(y)

        return y
