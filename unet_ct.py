"""
U-Net for sparse-view CT artifact removal (image-domain post-processing).

The network is trained to predict the RESIDUAL (the artifact pattern) rather
than the clean image directly, and that residual is subtracted from the
input. This is a common trick (used in FBPConvNet) that makes training
easier, because the residual is mostly close to zero and the network only
has to learn "what to remove" rather than reconstructing the whole image
from scratch.
"""

import torch
import torch.nn as nn


# Building block: two conv layers + batchnorm + ReLU, used at every U-Net level

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2

    Two convolutions in a row are standard in U-Net: one alone has small
    receptive field and limited representational power; stacking two 3x3
    convs approximates a larger effective receptive field while keeping the
    parameter count low.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# Full U-Net

class UNet(nn.Module):
    """
    Standard U-Net with 4 downsampling / 4 upsampling stages.

    in_channels=1, out_channels=1 because CT slices are single-channel, not RGB.

    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        c = base_channels

        # --- Encoder (downsampling path) ---
        # Each stage doubles channel count and halves spatial size.
        # growing receptive field lets the network "see" a
        # streak that spans the whole image, not just local patch.
        self.enc1 = DoubleConv(in_channels, c)        # full resolution
        self.enc2 = DoubleConv(c, c * 2)              
        self.enc3 = DoubleConv(c * 2, c * 4)          
        self.enc4 = DoubleConv(c * 4, c * 8)           # 1/8 resolution
        self.pool = nn.MaxPool2d(kernel_size=2)

        # --- Bottleneck (smallest spatial size, largest channel depth) ---
        self.bottleneck = DoubleConv(c * 8, c * 16)    # 1/16 resolution

        # --- Decoder (upsampling path) ---
        # ConvTranspose2d upsamples; after each upsample we concatenate the
        # matching encoder feature map (the "skip connection"). This is the
        # key ingredient that lets the network restore fine edge detail that
        # would otherwise be lost by downsampling into the bottleneck.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)          # input: c*8 (skip) + c*8 (up)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        # Final 1x1 conv maps back down to a single output channel
        # (the predicted residual / artifact map).
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool(e4))

        # ---- Decoder with skip connections ----
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)    # skip connection from e4
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)    # skip connection from e3
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)    # skip connection from e2
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)    # skip connection from e1
        d1 = self.dec1(d1)

        residual = self.out_conv(d1)

        # Residual learning: predict the artifact pattern, subtract it from
        # the (streaky) input to get the cleaned image.
        cleaned = x - residual
        return cleaned


# Minimal training loop sketch
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    One training epoch.

    dataloader is expected to yield (sparse_fbp_image, full_view_fbp_image)
    pairs, both as tensors of shape (batch, 1, H, W).
    """
    model.train()
    running_loss = 0.0

    for sparse_img, target_img in dataloader:
        sparse_img = sparse_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()
        prediction = model(sparse_img)
        loss = loss_fn(prediction, target_img)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sparse_img.size(0)

    return running_loss / len(dataloader.dataset)

