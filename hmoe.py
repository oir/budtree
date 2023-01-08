from typing import Optional

import torch
from torch import nn


class Hmoe(nn.Module):
    w: torch.Tensor
    b: torch.Tensor
    responses: torch.Tensor

    depth: int

    def __init__(self, depth: int, dimx: int, dimy: int):
        super().__init__()
        leaves = 2 ** (depth - 1)
        self.w = nn.Parameter(0.1 * torch.rand([dimx, leaves - 1]))
        self.b = nn.Parameter(torch.zeros([leaves - 1]))
        self.responses = nn.Parameter(0.1 * torch.rand([leaves, dimy]))
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[0]

        gate = torch.sigmoid(torch.matmul(x, self.w) + self.b)
        running_gating = torch.ones([bs, 1])

        for i in range(1, self.depth):
            parents = gate[:, (2 ** (i - 1) - 1) : (2**i - 1)]
            left = parents * running_gating
            right = (1.0 - parents) * running_gating
            # instead of concatenating children as [llllrrrr], we need to
            # concat as [lrlrlrlr] since that's how nodes are logically indexed:
            running_gating = torch.stack([left, right], dim=1).reshape(
                [bs, left.size()[1] + right.size()[1]]
            )

        return torch.matmul(running_gating, self.responses)


if __name__ == "__main__":
    import numpy as np
    from torch import nn

    import pickle
    import sys

    tra = np.loadtxt(sys.argv[1], delimiter=",")
    tst = np.loadtxt(sys.argv[2], delimiter=",")

    bs = 64
    model = nn.Sequential(
        nn.Conv2d(1, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Flatten(start_dim=1),
        Hmoe(8, 512, 10),
    )

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(model: nn.Module, data, train: bool):
        N = (np.shape(data)[0] + bs - 1) // bs
        perm = np.random.permutation(N) if train else range(N)
        total_loss = 0.0
        total_correct = 0
        total = 0

        for count in range(N):
            i = perm[count]
            start = i * bs
            bs_ = min(bs, np.shape(data)[0] - start)
            end = start + bs_

            y = torch.tensor(data[start:end, 0], dtype=torch.long)
            x = torch.tensor(data[start:end, 1:] / 255.0, dtype=torch.float)

            x = x.view([bs_, 28, 28]).unsqueeze(1)
            logit = model.forward(x)
            l = loss(logit, y)

            total_correct += (torch.argmax(logit, dim=1) == y).sum().item()
            total_loss += l.item() * bs_
            total += bs_

            if train:
                opt.zero_grad()
                l.backward()
                opt.step()

        return total_loss / total, 1.0 - total_correct / total

    epochs = 20

    for e in range(epochs):
        l, err = run_epoch(model, tra, train=True)
        lt, errt = run_epoch(model, tst, train=False)

        print(f"{l:.2f}  {err*100:>5.2f}  {lt:.2f}  {errt*100:>5.2f}")
