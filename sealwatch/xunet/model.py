import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.kv_filter = (
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ]
            ).view(1, 1, 5, 5)
            / 12.0
        )

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""
        kv_filter = self.kv_filter.to(inp.device)  # Move filter to the correct device
        return F.conv2d(inp, kv_filter, stride=1, padding=2)


class ConvBlock(nn.Module):
    """This class returns a building block for XuNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        use_abs: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if kernel_size > 1 else 0

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.use_abs = use_abs
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv -> batch_norm -> activation -> pooling."""
        x = self.conv(inp)
        x = self.batch_norm(x)
        if self.use_abs:
            x = torch.abs(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class XuNet(nn.Module):
    """This class returns the XuNet model."""

    def __init__(self) -> None:
        super().__init__()
        self.image_processing = ImageProcessing()
        self.layer1 = ConvBlock(1, 8, kernel_size=5, activation="tanh", use_abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Returns logits for the given tensor."""
        x = self.image_processing(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x