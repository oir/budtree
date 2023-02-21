from typing import Dict, List
import sys

import numpy as np
import imageio.v3 as iio
import torch
from torch import nn

from hmoe import Hmoe


def read_celeba(
    folder: str, partition: Dict[str, int], counts: List[int]
) -> np.ndarray:
    data = [
        np.ndarray([counts[i], 3, 218, 178], dtype="uint8") for i in range(len(counts))
    ]

    indices = [0, 0, 0]
    for i in range(1, sum(counts.values()) + 1):
        fname = f"{folder}/{i:06d}.jpg"
        part = partition[f"{i:06d}.jpg"]
        im = iio.imread(fname)
        data[part][indices[part], :, :, :] = np.transpose(im, [2, 0, 1])
        indices[part] += 1

    return data[0], data[1], data[2]


def read_partition(fname: str) -> Dict[str, int]:
    d = {}
    counts = {0: 0, 1: 0, 2: 0}
    with open(fname) as f:
        f.readline()  # first line is header
        for line in f:
            k, v = line.strip().split(",")
            v = int(v)
            d[k] = v
            counts[v] += 1

    return d, counts


def read_labels(fname: str, partition: Dict[str, int]) -> np.ndarray:
    labels = [[], [], []]

    with open(fname) as f:
        names = f.readline().strip().split(",")[1:]
        for line in f:
            items = line.strip().split(",")
            fname = items[0]
            vals = [int(int(x) > 0) for x in items[1:]]
            part = partition[fname]
            labels[part].append(vals)

    return np.array(labels[0]), np.array(labels[1]), np.array(labels[2]), names


if __name__ == "__main__":
    folder = "/home/oirsoy/data/celeba"
    partition, counts = read_partition(f"{folder}/list_eval_partition.csv")
    tra, dev, tst = read_celeba(
        f"{folder}/img_align_celeba/img_align_celeba/", partition, counts
    )
    ytra, ydev, ytst, names = read_labels(f"{folder}/list_attr_celeba.csv", partition)
    bs = 64
    epochs = 1

    # 15: eyeglasses, 20: male, 31: smiling, 35: wearing hat, 38: wearing necktie
    subset = [15, 20, 31, 35, 38]
    ytra = ytra[:, subset]
    ydev = ydev[:, subset]
    ytst = ytst[:, subset]

    model = nn.Sequential(
        nn.Conv2d(3, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(6, 4),
        nn.Conv2d(64, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(6, 4),
        nn.Flatten(start_dim=1),
        Hmoe(8, 3456, 5),
    )

    loss = torch.nn.BCELoss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.save(model, "model0")
    torch.save(opt, "opt0")

    def run_epoch(model: nn.Module, data_x, data_y, train: bool):
        N = (np.shape(data_x)[0] + bs - 1) // bs
        perm = np.random.permutation(N) if train else range(N)
        total_loss = 0.0
        total_correct = 0
        total = 0

        for count in range(N):
            print(count, N)
            i = perm[count]
            start = i * bs
            bs_ = min(bs, np.shape(data_x)[0] - start)
            end = start + bs_

            y = torch.tensor(data_y[start:end, :], dtype=torch.float)
            x = torch.tensor(data_x[start:end, :, :, :] / 255.0, dtype=torch.float)

            logit = model.forward(x)
            l = loss(logit, y)

            total_correct += ((logit > 0.0) == y).sum().item()
            total_loss += l.item() * bs_
            total += bs_

            print(total_loss / total)

            if train:
                opt.zero_grad()
                l.backward()
                opt.step()

        return total_loss / total, 1.0 - total_correct / total

    run_epoch(model, tra, ytra, train=True)
    torch.save(model, "model1")
    torch.save(opt, "opt1")
