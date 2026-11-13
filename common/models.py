"""Small models shared across this project. Deliberately small yet representative, so every node is easy to trace by eye on the graph."""

import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    """The simplest conv + relu, with no BN. Used to see the most basic observer placement."""

    def __init__(self, cin=3, cout=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBnReLU(nn.Module):
    """conv + bn + relu, the star of M04 fusion."""

    def __init__(self, cin=3, cout=8, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """A block with a skip connection. The two inputs of add must share qparams -> M03 SharedQuantizationSpec.

    The default ch=3 lets it take common.models.example_inputs() directly for standalone tests.
    """

    def __init__(self, ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class TinyConvNet(nn.Module):
    """A small classifier that can actually be trained, for M05 / M06.

    Input (N, 3, 16, 16); output num_classes logits.
    """

    def __init__(self, num_classes=4, width=16):
        super().__init__()
        self.stem = ConvBnReLU(3, width)
        self.pool1 = nn.MaxPool2d(2)          # 16 -> 8
        self.block = ResidualBlock(width)
        self.pool2 = nn.MaxPool2d(2)          # 8 -> 4
        self.head_conv = ConvBnReLU(width, width * 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        x = self.pool1(self.stem(x))
        x = self.pool2(self.block(x))
        x = self.head_conv(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


def example_inputs(batch_size=2, size=16):
    """The example input shared by all modules (needed by torch.export)."""
    return (torch.randn(batch_size, 3, size, size),)


MODELS = {
    "conv_relu": ConvReLU,
    "conv_bn_relu": ConvBnReLU,
    "residual": ResidualBlock,
    "tiny": TinyConvNet,
}


def get_model(name="conv_bn_relu", **kwargs):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}, available: {list(MODELS)}")
    return MODELS[name](**kwargs)


class CatPoolNet(nn.Module):
    """A small network with cat / maxpool / add.

    Used to observe which ops are forced to share qparams (SharedQuantizationSpec)
    and which ops quantize each of their inputs separately.
    """

    def __init__(self, ch=8):
        super().__init__()
        self.a = nn.Conv2d(3, ch, 3, padding=1)
        self.b = nn.Conv2d(3, ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, x):
        ya, yb = torch.relu(self.a(x)), torch.relu(self.b(x))
        z = torch.cat([ya, yb], dim=1)   # cat: all inputs and the output share qparams
        z = self.pool(z)                 # maxpool: input and output share qparams
        z = self.out(z)
        return z + z.mean()              # add: the two inputs are quantized separately


MODELS["cat_pool"] = CatPoolNet


class OpZooNet(nn.Module):
    """One model packed with many kinds of ops (M10-3): see how ops outside the list split the graph into segments.

    TinyConvNet only has conv / relu / maxpool / add / mean / linear, almost all on the backend's op list;
    this adds sigmoid, leaky_relu, cat, mul, avg_pool, matmul, softmax. No training needed; only the graph matters.
    """

    def __init__(self, ch=8, num_classes=10):
        super().__init__()
        self.conv_a = nn.Conv2d(3, ch, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(ch)
        self.conv_b = nn.Conv2d(3, ch, 1)
        self.proj = nn.Parameter(torch.randn(2 * ch, 2 * ch) / (2 * ch) ** 0.5)
        self.fc = nn.Linear(2 * ch, num_classes)

    def forward(self, x):
        a = torch.sigmoid(self.bn_a(self.conv_a(x)))               # Conv → BN → Sigmoid
        b = nn.functional.leaky_relu(self.conv_b(x))               # Conv → LeakyRelu
        z = torch.cat([a, b], dim=1) * 0.5                         # Concat, Mul
        z = nn.functional.avg_pool2d(nn.functional.max_pool2d(z, 2), 2)  # MaxPool, AveragePool
        z = z.mean(dim=(2, 3)) @ self.proj                         # ReduceMean, MatMul
        return torch.softmax(self.fc(z), dim=-1)                   # Gemm, Softmax


MODELS["op_zoo"] = OpZooNet
