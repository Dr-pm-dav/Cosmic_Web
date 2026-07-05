"""3D U-Net models: web segmentation and multifield reconstruction."""
import torch
import torch.nn as nn


def double_conv(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """3D U-Net mapping `in_channels` to `n_classes` per-voxel logits."""

    def __init__(self, in_channels=1, n_classes=4, f=16):
        super().__init__()
        self.e1 = double_conv(in_channels, f)
        self.e2 = double_conv(f, 2 * f)
        self.b = double_conv(2 * f, 4 * f)
        self.u2 = nn.ConvTranspose3d(4 * f, 2 * f, 2, 2)
        self.d2 = double_conv(4 * f, 2 * f)
        self.u1 = nn.ConvTranspose3d(2 * f, f, 2, 2)
        self.d1 = double_conv(2 * f, f)
        self.out = nn.Conv3d(f, n_classes, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class MultiFieldUNet(nn.Module):
    """Shared 3D U-Net backbone with a web-segmentation head plus one regression head per target field.

    forward(x) -> (web_logits, {target_name: prediction})
    """

    def __init__(self, in_channels=1, target_names=(), n_web=4, f=16):
        super().__init__()
        self.e1 = double_conv(in_channels, f)
        self.e2 = double_conv(f, 2 * f)
        self.b = double_conv(2 * f, 4 * f)
        self.u2 = nn.ConvTranspose3d(4 * f, 2 * f, 2, 2)
        self.d2 = double_conv(4 * f, 2 * f)
        self.u1 = nn.ConvTranspose3d(2 * f, f, 2, 2)
        self.d1 = double_conv(2 * f, f)
        self.pool = nn.MaxPool3d(2)
        self.web_head = nn.Conv3d(f, n_web, 1)
        self.reg_heads = nn.ModuleDict({n: nn.Conv3d(f, 1, 1) for n in target_names})

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        feat = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.web_head(feat), {n: h(feat) for n, h in self.reg_heads.items()}
