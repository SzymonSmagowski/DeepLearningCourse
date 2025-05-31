import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange, repeat


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic residual block with GroupNorm."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
        else:
            self.time_mlp = None
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)


class Attention(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = rearrange(qkv, 'b (three h d) x y -> three b h (x y) d', 
                           three=3, h=self.num_heads)
        
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        return x + self.proj(out)


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion models."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        channel_mult: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        use_scale_shift_norm: bool = True,
        image_size: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.image_size = image_size
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Initial projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [Block(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(Attention(ch, num_heads))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_channels.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.Sequential(
            Block(ch, ch, time_embed_dim),
            Attention(ch, num_heads),
            Block(ch, ch, time_embed_dim),
        )
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                layers = [Block(ch + skip_ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(Attention(ch, num_heads))
                
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Final projection
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_proj = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Downsampling
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, Block):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, Block):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, Block):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Final projection
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_proj(h)
        
        return h