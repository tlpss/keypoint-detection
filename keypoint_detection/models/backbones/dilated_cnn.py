import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone


class DilatedCnn(Backbone):
    def __init__(self, n_channels=32, **kwargs):
        super().__init__()
        self.n_channels_in = 3
        self.n_channels = n_channels
        kernel_size = (3, 3)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels_in,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_n_channels_out(self):
        return self.n_channels
