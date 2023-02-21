from budtree.hmoe import Hmoe
from budtree.utilities import ensure_file
import numpy as np
import torch
from torch import nn
import typer

import pickle
import sys
from enum import Enum

TRAIN_URL = "https://pjreddie.com/media/files/mnist_train.csv"
TEST_URL = "https://pjreddie.com/media/files/mnist_test.csv"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


def architecture() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Flatten(start_dim=1),
        Hmoe(8, 512, 10),
    )


def main(epochs: int = 10, batch_size: int = 64, device: Device = Device.auto):
    dev = (
        "cuda:0"
        if (
            device == Device.gpu
            or (device == Device.auto and torch.cuda.is_available())
        )
        else "cpu"
    )

    tra_fname = ensure_file("mnist_train.csv", TRAIN_URL)
    tst_fname = ensure_file("mnist_test.csv", TEST_URL)

    tra = np.loadtxt(tra_fname, delimiter=",")
    tst = np.loadtxt(tst_fname, delimiter=",")

    model = architecture().to(dev)
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(model: nn.Module, data, train: bool):
        N = (np.shape(data)[0] + batch_size - 1) // batch_size
        perm = np.random.permutation(N) if train else range(N)
        total_loss = 0.0
        total_correct = 0
        total = 0

        for count in range(N):
            i = perm[count]
            start = i * batch_size
            batch_size_ = min(batch_size, np.shape(data)[0] - start)
            end = start + batch_size_

            y = torch.tensor(data[start:end, 0], dtype=torch.long).to(dev)
            x = torch.tensor(data[start:end, 1:] / 255.0, dtype=torch.float).to(dev)

            # instead of a bs x 784 shape or flat vectors, we need bs x 28 x 28
            # of 2D images. we also need a "number of channels" of 1, which is
            # what the .unsqueeze adds.
            x = x.view([batch_size_, 28, 28]).unsqueeze(1)

            logit = model.forward(x)
            l = loss(logit, y)

            total_correct += (torch.argmax(logit, dim=1) == y).sum().item()
            total_loss += l.item() * batch_size_
            total += batch_size_

            if train:
                opt.zero_grad()
                l.backward()
                opt.step()

        return total_loss / total, 1.0 - total_correct / total

    # underline a string in supporting terminals
    def ul(s: str) -> str:
        return f"\x1b[4m{s}\x1b[24m"

    print(ul("train      ") + "  " + ul("test       "))
    print(ul("loss") + "  " + ul("%-err") + "  " + ul("loss") + "  " + ul("%-err"))

    for e in range(epochs):
        l, err = run_epoch(model, tra, train=True)
        lt, errt = run_epoch(model, tst, train=False)

        print(f"{l:.2f}  {err*100:>5.2f}  {lt:.2f}  {errt*100:>5.2f}")


if __name__ == "__main__":
    typer.run(main)
