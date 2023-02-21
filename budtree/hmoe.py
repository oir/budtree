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
        running_gating = torch.ones([bs, 1], device=x.device)

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
