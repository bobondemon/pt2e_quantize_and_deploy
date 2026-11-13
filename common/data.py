"""Synthetic dataset: nothing to download, so training and calibration work offline.

Task: a 16x16 RGB image contains one bright patch; decide which quadrant it is in (4 classes).
Simple enough that TinyConvNet learns it in a few dozen steps, yet with enough activation
dynamic range for quantization error to show up.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

SIZE = 16
NUM_CLASSES = 4


def make_tensors(n=512, seed=0, noise=0.8, outlier_ratio=0.0):
    """Return (images, labels).

    When outlier_ratio > 0, some pixels are randomly blown up to an extreme value,
    which M05 uses to show MinMaxObserver being blown out by outliers.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, SIZE, SIZE, generator=g) * noise
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    half = SIZE // 2
    for i in range(n):
        q = int(y[i])
        r0 = (q // 2) * half
        c0 = (q % 2) * half
        # Place a 4x4 bright patch at a random position inside that quadrant
        dr = int(torch.randint(0, half - 3, (1,), generator=g))
        dc = int(torch.randint(0, half - 3, (1,), generator=g))
        ch = int(torch.randint(0, 3, (1,), generator=g))
        x[i, ch, r0 + dr : r0 + dr + 4, c0 + dc : c0 + dc + 4] += 0.9

    if outlier_ratio > 0:
        k = max(1, int(n * outlier_ratio))
        idx = torch.randperm(n, generator=g)[:k]
        x[idx, 0, 0, 0] = 50.0  # exaggerated outlier

    return x, y


def make_loader(n=512, batch_size=32, seed=0, shuffle=True, **kw):
    x, y = make_tensors(n=n, seed=seed, **kw)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def calibration_batches(n=128, batch_size=16, seed=123, **kw):
    """Iterator for PTQ calibration (yields inputs only; no labels needed)."""
    x, _ = make_tensors(n=n, seed=seed, **kw)
    for i in range(0, n, batch_size):
        yield (x[i : i + batch_size],)


@torch.no_grad()
def evaluate(model, loader):
    """Return top-1 accuracy."""
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, loader):
    """Return the mean cross-entropy loss. The main metric from M06 on: smoother and more sensitive than accuracy, and QAT optimizes it directly."""
    lossf = torch.nn.CrossEntropyLoss(reduction="sum")
    total = count = 0.0
    for x, y in loader:
        total += float(lossf(model(x), y))
        count += int(y.numel())
    return total / max(count, 1)


def train(model, loader, epochs=3, lr=1e-2, log_every=0):
    """Minimal training loop (shared by fp32 and QAT)."""
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
    """A trained TinyConvNet (eval mode). Shared by the scripts from M05 on; the seed is fixed so results are reproducible.

    After 20 epochs it reaches about 0.71 accuracy on this synthetic task: not high, but low enough that the difference before and after quantization is visible.
    """
    from common.models import TinyConvNet

    torch.manual_seed(seed)
    model = TinyConvNet()
    train(model, make_loader(n=1024, seed=1), epochs=epochs)
    return model.eval()
