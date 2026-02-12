import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        self.dim = dim
        # 
        self.kernel_3 = nn.Parameter(torch.ones(dim, 1, 3, 3))      # dilation=1
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))   # dilation=3
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))   # dilation=5
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, h):
        # 
        h1 = F.conv2d(h, self.kernel_3, padding=1, dilation=1, groups=self.dim)
        h2 = F.conv2d(h, self.kernel_3_1, padding=3, dilation=3, groups=self.dim)
        h3 = F.conv2d(h, self.kernel_3_2, padding=5, dilation=5, groups=self.dim)
        out = self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size,dim,height,width = 2,64,13,13

    model = MSCA(dim=dim)

    # [B, C, H, W]
    x = torch.randn(batch_size, dim, height, width)

    print(f"Input shape: {x.shape}")

    out = model(x)
    print(f"Output shape: {out.shape}")
