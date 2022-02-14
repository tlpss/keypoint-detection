import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone


class ResNetBlock(nn.Module):
    """
    based on the basic ResNet Block used in torchvision
    inspired on https://jarvislabs.ai/blogs/resnet
    """

    def __init__(self, n_channels_in, n_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class S3K(Backbone):
    """
    Backbone (approx) as in the S3K paper by Mel
    inspired by Peter's version of the backbone.
    """

    def __init__(self):
        self.kernel_size = (3, 3)
        super(S3K, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=self.kernel_size, padding="same")
        self.norm1 = nn.BatchNorm2d(num_features=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, stride=(2, 2))
        self.norm2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2))
        self.norm3 = nn.BatchNorm2d(num_features=32)
        self.res1 = ResNetBlock(32)
        self.res2 = ResNetBlock(32)
        self.res3 = ResNetBlock(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding="same")
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2))

        self.conv5 = nn.Conv2d(
            in_channels=32 + 16, out_channels=32, kernel_size=self.kernel_size, padding="same", bias=False
        )
        self.norm4 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=(2, 2), output_padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=32 + 3, out_channels=32, kernel_size=self.kernel_size, padding="same", bias=False
        )
        self.norm5 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x_0 = x
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x_1 = x
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.res1(x)

        x = self.res2(x)

        x = self.res3(x)

        x = self.conv4(x)

        x = self.up1(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.conv5(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.up2(x)
        x = torch.cat([x, x_0], dim=1)
        x = self.conv6(x)
        x = self.norm5(x)
        x = self.relu(x)
        return x

    def get_n_channels_out(self):
        return 32
