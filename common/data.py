"""合成資料集：不用下載任何東西，離線就能訓練/校準。

任務：16x16 的 RGB 影像上有一個亮塊，判斷它在哪一個象限（4 類）。
簡單到 TinyConvNet 幾十個 step 就能學會，但又有足夠的 activation 動態範圍
可以讓量化誤差顯現出來。
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

SIZE = 16
NUM_CLASSES = 4


def make_tensors(n=512, seed=0, noise=0.8, outlier_ratio=0.0):
    """回傳 (images, labels)。

    outlier_ratio > 0 時會隨機把一些像素放大成極端值，
    用來在 M05 展示 MinMaxObserver 被 outlier 拉爆的現象。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, SIZE, SIZE, generator=g) * noise
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    half = SIZE // 2
    for i in range(n):
        q = int(y[i])
        r0 = (q // 2) * half
        c0 = (q % 2) * half
        # 在該象限內隨機放一個 4x4 的亮塊
        dr = int(torch.randint(0, half - 3, (1,), generator=g))
        dc = int(torch.randint(0, half - 3, (1,), generator=g))
        ch = int(torch.randint(0, 3, (1,), generator=g))
        x[i, ch, r0 + dr : r0 + dr + 4, c0 + dc : c0 + dc + 4] += 0.9

    if outlier_ratio > 0:
        k = max(1, int(n * outlier_ratio))
        idx = torch.randperm(n, generator=g)[:k]
        x[idx, 0, 0, 0] = 50.0  # 誇張的 outlier

    return x, y


def make_loader(n=512, batch_size=32, seed=0, shuffle=True, **kw):
    x, y = make_tensors(n=n, seed=seed, **kw)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def calibration_batches(n=128, batch_size=16, seed=123, **kw):
    """PTQ 校準用的 iterator（只回傳 input，不需要 label）。"""
    x, _ = make_tensors(n=n, seed=seed, **kw)
    for i in range(0, n, batch_size):
        yield (x[i : i + batch_size],)


@torch.no_grad()
def evaluate(model, loader):
    """回傳 top-1 accuracy。"""
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, loader):
    """回傳平均 cross-entropy loss。M06 起的主指標：比 accuracy 連續、敏感，且 QAT 直接最佳化它。"""
    lossf = torch.nn.CrossEntropyLoss(reduction="sum")
    total = count = 0.0
    for x, y in loader:
        total += float(lossf(model(x), y))
        count += int(y.numel())
    return total / max(count, 1)


def train(model, loader, epochs=3, lr=1e-2, log_every=0):
    """極簡訓練迴圈（fp32 與 QAT 共用）。"""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = torch.nn.CrossEntropyLoss()
    step = 0
    for ep in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"  ep{ep} step{step} loss={loss.item():.4f}")
    return model


def trained_tiny_net(epochs=20, seed=0):
    """訓練好的 TinyConvNet（eval 模式）。M05 起的腳本共用，固定 seed 讓結果可重現。

    20 epoch 後在這個合成任務上約 0.71 accuracy：不算高，但量化前後的差異才看得出來。
    """
    from common.models import TinyConvNet

    torch.manual_seed(seed)
    model = TinyConvNet()
    train(model, make_loader(n=1024, seed=1), epochs=epochs)
    return model.eval()
