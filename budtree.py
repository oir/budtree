from typing import Optional

import torch


class Node:
    _inherit: bool = True
    _lr: float = 0.1  # TODO: more sophisticated (or generic) optimizers

    w: torch.Tensor
    b: torch.Tensor
    leafness: torch.Tensor  # γ in paper
    response: torch.Tensor  # ρ in paper

    left: Optional["Node"]
    right: Optional["Node"]

    def __init__(self, dimx: int, dimy: int):
        self.w = torch.rand([dimx], requires_grad=True)
        self.w.data -= 0.5
        self.w.data *= 0.1
        self.b = torch.zeros([], requires_grad=True)
        self.leafness = torch.ones([], requires_grad=True)
        self.response = torch.rand([dimy], requires_grad=True)
        self.response.data *= 0.1

        self.left = self.right = None

    def _split(self):
        dimx = self.w.size()[0]
        dimy = self.response.size()[0]

        self.left = Node(dimx, dimy)
        self.right = Node(dimx, dimy)
        if self._inherit:
            self.left.w.data += self.w.data
            self.right.w.data += self.w.data
            self.left.b.data += self.b.data
            self.right.b.data += self.b.data

    def leaf(self) -> bool:
        return self.leafness == 1.0 or self.left is None

    def physical_leaf(self) -> bool:
        return self.left is None

    def size(self) -> int:
        if self.leaf():
            return 1
        return 1 + self.left.size() + self.right.size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[0]

        response = self.leafness * self.response.unsqueeze(0).expand([bs, -1])

        if self.leaf():
            return response

        dimy = response.size()[1]

        gate = torch.sigmoid(torch.matmul(x, self.w) + self.b)
        gate = gate.unsqueeze(1).expand([-1, dimy])
        return response + (1.0 - self.leafness) * (
            self.left.forward(x) * gate + self.right.forward(x) * (1.0 - gate)
        )

    def reset_grad(self):
        self.w.grad = None
        self.b.grad = None
        self.leafness.grad = None
        self.response.grad = None

        if not self.leaf():
            self.left.reset_grad()
            self.right.reset_grad()

    def _update_param(self, w: torch.Tensor):
        if w.grad is not None:
            w.data -= self._lr * w.grad

    def update(self):
        self._update_param(self.w)
        self._update_param(self.b)
        self._update_param(self.leafness)
        self._update_param(self.response)

        if self.leafness.data.item() > 1:
            self.leafness.data = torch.ones([])

        if not self.leaf():
            self.left.update()
            self.right.update()

        if self.physical_leaf() and self.leafness.data.item() < 1:
            self._split()


if __name__ == "__main__":
    import numpy as np

    import pickle
    import sys

    tra = np.loadtxt(sys.argv[1], delimiter=",")
    tst = np.loadtxt(sys.argv[2], delimiter=",")

    bs = 64
    n = Node(784, 10)
    loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def run_epoch(tree: Node, data, train: bool):
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

            logit = tree.forward(x)
            l = loss(logit, y)

            total_correct += (torch.argmax(logit, dim=1) == y).sum().item()
            total_loss += l.item() * bs_
            total += bs_

            if train:
                l.backward()
                tree.update()
                tree.reset_grad()

        return total_loss / total, 1.0 - total_correct / total

    epochs = 10

    for e in range(epochs):
        l, err = run_epoch(n, tra, train=True)
        lt, errt = run_epoch(n, tst, train=False)

        print(f"{l:.2f}  {err*100:>5.2f}  {lt:.2f}  {errt*100:>5.2f}  {n.size()}")

    model = open("tmp.model", "wb")
    pickle.dump(n, model)
    model.close()

    model = open("tmp.model", "rb")
    n2 = pickle.load(model)
    model.close()

    l, err = run_epoch(n2, tra, train=False)
    lt, errt = run_epoch(n2, tst, train=False)

    print(f"{l:.2f}  {err*100:>5.2f}  {lt:.2f}  {errt*100:>5.2f}  {n.size()}")
