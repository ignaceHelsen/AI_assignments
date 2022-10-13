import torch
import torch.nn as nn
from collections import OrderedDict


class Generator(nn.Module):
    """Your implementation of the generator of DCGAN"""

    def __init__(self, config: dict):
        """TODO: define all layers of the Generator."""
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16),
            nn.Tanh()
        )

    def forward(self, input: torch.tensor) -> torch.Tensor:
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    """Your implementation of the discriminator of DCGAN"""

    def __init__(self):
        """TODO: define all layers of the Discriminator."""
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.main(input)
        return output
