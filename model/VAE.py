import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, nc=1, ch_e=64, ch_d=64, nz=64):
        super().__init__()
        self.nc = nc
        self.ch_e = ch_e
        self.ch_d = ch_d
        self.nz = nz

        self.encoder = Encoder(nc, ch_e, nz)
        self.decoder = Decoder(nc, ch_d, nz)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z)

        return recon_x, mean, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, nc=1, ch=64, nz=64):
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.nz = nz

        self.conv = nn.Sequential(
            nn.Conv2d(nc, ch, kernel_size=4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),   # 128x128 -> 64x64
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, ch * 16, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(ch * 16),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.linear_mean = nn.Sequential(
            nn.Linear(ch * 16 * 8 * 8, nz),
        )

        self.linear_log_var = nn.Sequential(
            nn.Linear(ch * 16 * 8 * 8, nz),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.ch * 16 * 8 * 8)
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, nc=1, ch=64, nz=48):
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.nz = nz

        self.linear = nn.Sequential(
                nn.Linear(nz, ch * 16 * 8 * 8)
            )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.ConvTranspose2d(ch, nc, kernel_size=4, stride=2, padding=1),  # 128x128 -> 256x256
            nn.Tanh()
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, self.ch * 16, 8, 8)
        out = self.deconv(out)
        return out

