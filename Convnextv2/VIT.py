import torch
import torch.nn as nn

#GWIM: gate-weighted interaction module
class react(nn.Module):
    def __init__(self, dim, reduction=1):
        super(react, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=True),
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=True),
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)

        x11 = self.conv1(x)
        x22 = self.conv2(x)

        xxx = x11 * x22

        avg = self.avg_pool(xxx).view(B, self.dim)
        avg = avg.reshape(B, self.dim, 1, 1)

        avg_ = 1 - avg
        avg_ = avg_.reshape(B, self.dim, 1, 1)

        x1 = avg * x1
        x2 = avg_ * x2

        return x1, x2


