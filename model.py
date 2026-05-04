# Video TAA + DLAA model
# MSXYZ


import torch
import torch.nn as nn


class DLAANet(nn.Module):
    """Small residual refinement model for the DLAA pass."""
    
    def __init__(self):
        super().__init__()

        base_channels = 192

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )
        
        self.register_buffer(
            "jitter_offsets",
            torch.tensor([
                [ 0.0000, -0.1667],
                [-0.2500,  0.1667],
                [ 0.2500, -0.3889],
                [-0.3750, -0.0556],
                [ 0.1250,  0.2778],
                [-0.1250, -0.2778],
                [ 0.3750,  0.0556],
                [-0.4375,  0.3889],
                [ 0.0625, -0.4630],
                [-0.3125,  0.1296],
                [ 0.1875, -0.2963],
                [-0.4375, -0.0370],
                [ 0.3125,  0.2407],
                [-0.0625, -0.2037],
                [ 0.4375,  0.0185],
                [-0.4688,  0.3148],
            ], dtype=torch.float32)
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.reconstructor = nn.Conv2d(64, 3, 3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
            elif isinstance(layer, nn.GroupNorm):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, image):
        input_img = image

        feat_low = self.enc1(input_img)
        feat_high = self.enc2(feat_low)

        context = self.bottleneck(feat_high)

        decoded = self.dec(torch.cat([context, feat_low], dim=1))
        residual = self.reconstructor(decoded)

        return torch.clamp(input_img + residual, 0.0, 1.0)