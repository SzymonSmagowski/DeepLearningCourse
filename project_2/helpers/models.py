import math
import torch
import torch.nn as nn
import torch.nn.init as init


class ConvBaseline(nn.Module):
    """
    Extremely small CNN => ~120 k params.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                              # (20, 50)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                              # (10, 25)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))                  # (128, 1, 1)
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x).flatten(1)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """
    Classic sine/cos positional encodings.
    Expects input shape (B, T, D).
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self,
                 n_classes,
                 n_mfcc=40,
                 d_model=128, n_heads=4,
                 num_layers=4, dim_ff=256,
                 dropout=0.1, pool="cls"):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.pos_enc    = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):                     # x: (B, 40, 101)
        x = x.transpose(1, 2)                 #      (B, 101, 40)
        x = self.input_proj(x)                #      (B, 101, d_model)

        if self.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)  # (B,1,d_model)
            x   = torch.cat([cls, x], dim=1)                # (B, 102, d_model)

        x = self.pos_enc(x)
        x = self.encoder(x)                   # (B, T, d_model)

        rep = x[:, 0] if self.pool == "cls" else x.mean(1)
        return self.classifier(rep)