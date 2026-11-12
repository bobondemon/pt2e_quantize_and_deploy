"""本專案共用的小模型。故意做得小又有代表性，方便在 graph 上肉眼追蹤每個 node。"""

import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    """最單純的 conv + relu，沒有 BN。用來看最基本的 observer 落點。"""

    def __init__(self, cin=3, cout=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBnReLU(nn.Module):
    """conv + bn + relu，M04 fusion 的主角。"""

    def __init__(self, cin=3, cout=8, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """帶 skip connection 的區塊。add 的兩個輸入需要共用 qparams -> M03 SharedQuantizationSpec。

    預設 ch=3 是為了能直接吃 common.models.example_inputs() 單獨測試。
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
    """能真的訓練起來的小分類網路，給 M05 / M06 用。

    輸入 (N, 3, 16, 16)，輸出 num_classes 個 logits。
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
    """所有模組共用的 example input（torch.export 需要）。"""
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
    """含 cat / maxpool / add 的小網路。

    用來觀察哪些 op 會被強制「共用 qparams」(SharedQuantizationSpec)，
    哪些 op 的每個輸入各自量化。
    """

    def __init__(self, ch=8):
        super().__init__()
        self.a = nn.Conv2d(3, ch, 3, padding=1)
        self.b = nn.Conv2d(3, ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, x):
        ya, yb = torch.relu(self.a(x)), torch.relu(self.b(x))
        z = torch.cat([ya, yb], dim=1)   # cat：所有輸入與輸出共用 qparams
        z = self.pool(z)                 # maxpool：輸入輸出共用 qparams
        z = self.out(z)
        return z + z.mean()              # add：兩個輸入各自量化


MODELS["cat_pool"] = CatPoolNet


class OpZooNet(nn.Module):
    """一個模型塞進多種 op（M10-3）：看清單外的 op 怎麼把圖切段。

    TinyConvNet 只有 conv／relu／maxpool／add／mean／linear，幾乎都在後端的 op 清單上；
    這裡再加 sigmoid、leaky_relu、cat、mul、avg_pool、matmul、softmax。不用訓練，只看圖。
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
        z = torch.cat([a, b], dim=1) * 0.5                         # Concat、Mul
        z = nn.functional.avg_pool2d(nn.functional.max_pool2d(z, 2), 2)  # MaxPool、AveragePool
        z = z.mean(dim=(2, 3)) @ self.proj                         # ReduceMean、MatMul
        return torch.softmax(self.fc(z), dim=-1)                   # Gemm、Softmax


MODELS["op_zoo"] = OpZooNet
