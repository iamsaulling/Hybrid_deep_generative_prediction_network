import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc=1, ch=64):
        super().__init__()
        self.nc = nc
        self.ch = ch

        self.conv1 = nn.Conv2d(nc, ch, kernel_size=4, stride=2, padding=1)
        self.conv1_in = nn.InstanceNorm2d(ch, affine=True)

        self.conv2 = nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
        self.conv2_in = nn.InstanceNorm2d(ch * 2, affine=True)

        self.conv3 = nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1)
        self.conv3_in = nn.InstanceNorm2d(ch * 4, affine=True)

        self.conv4 = nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1)
        self.conv4_in = nn.InstanceNorm2d(ch * 8, affine=True)

        self.conv5 = nn.Conv2d(ch * 8, ch * 16, kernel_size=4, stride=2, padding=1)
        self.conv5_in = nn.InstanceNorm2d(ch * 16, affine=True)

        self.fc = nn.Sequential(
            nn.Linear(ch * 16 * 8 * 8, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        # out_1 = x
        x = F.leaky_relu(self.conv1_in(x), 0.2, inplace=True)

        x = self.conv2(x)
        # out_2 = x
        x = F.leaky_relu(self.conv2_in(x), 0.2, inplace=True)

        x = self.conv3(x)
        # out_3 = x
        x = F.leaky_relu(self.conv3_in(x), 0.2, inplace=True)

        x = self.conv4(x)
        # out_4 = x
        x = F.leaky_relu(self.conv4_in(x), 0.2, inplace=True)

        x = self.conv5(x)
        # out_5 = x
        x = F.leaky_relu(self.conv5_in(x), 0.2, inplace=True)

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x