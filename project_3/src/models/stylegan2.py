import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from typing import List, Optional, Tuple
import math


class PixelNorm(nn.Module):
    """Pixel normalization layer."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_init: float = 0,
        lr_mul: float = 1,
        activation: Optional[str] = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * math.sqrt(2)
        else:
            out = F.linear(x, self.weight * self.scale)
        
        if self.bias is not None:
            out = out + self.bias * self.lr_mul
        
        return out


class EqualConv2d(nn.Module):
    """Conv2d layer with equalized learning rate."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )
        return out


class ModulatedConv2d(nn.Module):
    """Modulated convolution layer for StyleGAN2."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1],
        fused: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.fused = fused
        
        fan_in = in_channels * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        
        self.modulation = EqualLinear(style_dim, in_channels, bias_init=1)
        
        if upsample:
            self.blur = Blur(blur_kernel, pad=(len(blur_kernel) - 1) // 2, upsample_factor=2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, in_channels, height, width = x.shape
        
        # Modulation
        style = self.modulation(style).view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        
        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        
        if self.upsample:
            x = x.view(1, batch * in_channels, height, width)
            weight = weight.view(
                batch, self.out_channels, in_channels, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channels, self.out_channels, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)
            out = self.blur(out)
        else:
            x = x.view(1, batch * in_channels, height, width)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)
        
        return out


class StyledConv(nn.Module):
    """Styled convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1]
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel
        )
        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x, style)
        out = self.noise(out, noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB layer."""
    
    def __init__(self, in_channels: int, style_dim: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        
        if upsample:
            self.blur = Blur([1, 3, 3, 1], pad=1, upsample_factor=2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x, style)
        
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
                skip = self.blur(skip)
            
            # Ensure dimensions match
            if out.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            
            out = out + skip
        
        return out


class Blur(nn.Module):
    """Blur layer."""
    
    def __init__(self, kernel: List[int], pad: int, upsample_factor: int = 1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[None, None, :] * kernel[None, :, None]
        kernel = kernel / kernel.sum()
        
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        
        self.register_buffer('kernel', kernel)
        self.pad = pad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel.repeat(x.shape[1], 1, 1, 1), padding=self.pad, groups=x.shape[1])


class NoiseInjection(nn.Module):
    """Noise injection layer."""
    
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch, _, height, width = x.shape
            noise = x.new_empty(batch, 1, height, width).normal_()
        
        return x + self.weight * noise


class MappingNetwork(nn.Module):
    """Mapping network for StyleGAN2."""
    
    def __init__(
        self,
        z_dim: int,
        w_dim: int,
        num_ws: int,
        num_layers: int = 8,
        lr_multiplier: float = 0.01
    ):
        super().__init__()
        self.num_ws = num_ws
        
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(
                EqualLinear(
                    in_dim, w_dim, lr_mul=lr_multiplier, activation='fused_lrelu'
                )
            )
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.layers(z)
        return w.unsqueeze(1).repeat(1, self.num_ws, 1)


class Generator(nn.Module):
    """StyleGAN2 Generator."""
    
    def __init__(
        self,
        size: int,
        style_dim: int,
        n_mlp: int = 8,
        channel_multiplier: int = 2,
        blur_kernel: List[int] = [1, 3, 3, 1],
        lr_mlp: float = 0.01
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        
        self.style = nn.Sequential(*layers)
        
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        self.input = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.conv1 = StyledConv(512, 512, 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(512, style_dim, upsample=False)
        
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        
        in_channel = 512
        
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel
                )
            )
            
            self.convs.append(
                StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel)
            )
            
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            
            in_channel = out_channel
        
        self.n_latent = self.log_size * 2 - 2
        self.num_layers = self.n_latent
    
    def forward(
        self,
        styles: List[torch.Tensor],
        noise: Optional[List[torch.Tensor]] = None,
        randomize_noise: bool = True,
        return_latents: bool = False
    ) -> torch.Tensor:
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    torch.zeros(styles[0].shape[0], 1, 2 ** (2 + i // 2), 2 ** (2 + i // 2), device=styles[0].device)
                    for i in range(self.num_layers)
                ]
        
        if not isinstance(styles, list):
            styles = [styles] * self.n_latent
        
        latent = styles[0]
        
        out = self.input.repeat(latent.shape[0], 1, 1, 1)
        out = self.conv1(out, latent, noise[0])
        
        skip = self.to_rgb1(out, latent)
        
        i = 1
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, styles[i], noise[i])
            out = conv2(out, styles[i + 1], noise[i + 1])
            skip = to_rgb(out, styles[i + 2], skip)
            
            i += 2
        
        image = torch.tanh(skip)
        
        if return_latents:
            return image, latent
        
        return image


class Discriminator(nn.Module):
    """StyleGAN2 Discriminator."""
    
    def __init__(
        self,
        size: int,
        channel_multiplier: int = 2,
        blur_kernel: List[int] = [1, 3, 3, 1]
    ):
        super().__init__()
        
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        convs = [ConvLayer(3, channels[size], 1)]
        
        log_size = int(math.log(size, 2))
        
        in_channel = channels[size]
        
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            
            in_channel = out_channel
        
        self.convs = nn.Sequential(*convs)
        
        self.stddev_group = 4
        self.stddev_feat = 1
        
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convs(x)
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        
        # Minibatch standard deviation - simplified version
        # Calculate std for each example over all features
        stddev = out.reshape(batch, -1).std(dim=1, keepdim=True)
        # Expand to match spatial dimensions
        stddev = stddev.reshape(batch, 1, 1, 1).expand(batch, 1, height, width)
        
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        
        out = out.view(batch, -1)
        out = self.final_linear(out)
        
        return out


class ConvLayer(nn.Sequential):
    """Convolution layer with activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1],
        bias: bool = True,
        activate: bool = True
    ):
        layers = []
        
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        
        layers.append(
            EqualConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate
            )
        )
        
        if activate:
            layers.append(FusedLeakyReLU(out_channels, bias=bias))
        
        super().__init__(*layers)


class ResBlock(nn.Module):
    """Residual block for discriminator."""
    
    def __init__(self, in_channels: int, out_channels: int, blur_kernel: List[int] = [1, 3, 3, 1]):
        super().__init__()
        
        self.conv1 = ConvLayer(in_channels, in_channels, 3)
        self.conv2 = ConvLayer(in_channels, out_channels, 3, downsample=True)
        
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=True, activate=False, bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        
        return out


class FusedLeakyReLU(nn.Module):
    """Fused LeakyReLU with bias."""
    
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            out = F.leaky_relu(x + self.bias.view((1, -1) + (1,) * (x.dim() - 2)), 0.2)
        else:
            out = F.leaky_relu(x, 0.2)
        
        return out * math.sqrt(2)