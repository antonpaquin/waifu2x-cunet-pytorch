import math

import numpy as np
from PIL import Image
import torch
from torch import nn


class DIM:
    BATCH = 0
    CHANNEL = 1
    WIDTH = 2
    HEIGHT = 3


class UpCunet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.cunet_unet1 = CunetUnet1(channels, deconv=True)
        self.cunet_unet2 = CunetUnet2(channels, deconv=False)
        self.spatial_zero_padding = SpatialZeroPadding(-20)

    def forward(self, x):
        x = self.cunet_unet1.forward(x)
        x0 = self.cunet_unet2.forward(x)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        x = torch.clamp(x, min=0, max=1)
        return x
    
    def naive_upscale(self, pil_img: Image) -> Image:
        """
        Naive upscale: run the model through w2x and return a PIL image for the result.
        "Naive" because the nagadomi version breaks the image into tiles, and allows for running the model multiple
        times with some augmentations to improve the end result

        """
        x = np.asarray(pil_img)
        # upcunet drops 36 pixels as it convolves down the image.
        # Add some approximately similar padding; this will be stripped in the end result
        # Input is expected to have even dimensions. If not, pad an extra pixel and undo it at the end
        pad_h = x.shape[0] % 2
        pad_w = x.shape[1] % 2
        x = np.pad(x, [(18, 18 + pad_h), (18, 18 + pad_w), (0, 0)], mode='reflect')
        x = x.transpose([2, 0, 1])[np.newaxis, :, :, :]
        x = torch.Tensor(x / 255)
        y = self.forward(x).detach().numpy()
        y = (y * 255).astype('uint8')
        y = y[0].transpose(1, 2, 0)
        unpad = (slice(0, -2) if pad_h else slice(None), slice(0, -2) if pad_w else slice(None), slice(None))
        y = y[unpad]
        return Image.fromarray(y)

    
class CunetUnet1(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(64, 128, 64, se=True)
        self.unet_branch = UnetBranch(block1, 64, 64, depad=-4)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)
        
    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class CunetUnet2(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(128, 256, 128, se=True)
        block2 = nn.Sequential(
            UnetConv(64, 64, 128, se=True),
            UnetBranch(block1, 128, 128, depad=-4),
            UnetConv(128, 64, 64, se=True),
        )
        self.unet_branch = UnetBranch(block2, 64, 64, depad=-16)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)

    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class UnetConv(nn.Module):
    def __init__(self, channels_in: int, channels_mid: int, channels_out: int, se: bool):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_mid, 3)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(channels_mid, channels_out, 3)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.se = se
        if se:
            self.se_block = SEBlock(channels_out, r=8)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        if self.se:
            x = self.se_block(x)
        return x


class UnetBranch(nn.Module):
    def __init__(self, insert: nn.Module, channels_in: int, channels_out: int, depad: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_in, kernel_size=2, stride=2)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.insert = insert
        self.conv1 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.spatial_zero_padding = SpatialZeroPadding(depad)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.lrelu0(x0)
        x0 = self.insert(x0)
        x0 = self.conv1(x0)
        x0 = self.lrelu1(x0)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels_out: int, r: int):
        super().__init__()
        channels_mid = math.floor(channels_out / r)
        self.conv0 = nn.Conv2d(channels_out, channels_mid, kernel_size=1, stride=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_mid, channels_out, kernel_size=1, stride=1)
        self.sigmoid0 = nn.Sigmoid()

    def forward(self, x):
        x0 = torch.mean(x, dim=(DIM.WIDTH, DIM.HEIGHT), keepdim=True)
        x0 = self.conv0(x0)
        x0 = self.relu0(x0)
        x0 = self.conv1(x0)
        x0 = self.sigmoid0(x0)
        x = torch.mul(x, x0)
        return x


class SpatialZeroPadding(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        if padding > 0:
            raise NotImplementedError("I don't know how to actually pad 0s")
        self.slice = [slice(None) for _ in range(4)]
        self.slice[DIM.HEIGHT] = slice(-padding, padding)
        self.slice[DIM.WIDTH] = slice(-padding, padding)

    def forward(self, x):
        return x[self.slice]
