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
    

# Add this class to your existing models.py file

class SimpleRNN(nn.Module):
    """
    Simple RNN model for speech command classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.0):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output size will be doubled if using bidirectional RNN
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layer
        self.fc = nn.Linear(fc_size, num_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = x.transpose(1, 2)  # from (batch, features, time) to (batch, time, features)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Use the final time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.fc(out)
        return out
    

class GRUModel(nn.Module):
    """GRU model for speech command classification"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer size depends on bidirectional setting
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layer
        self.fc = nn.Linear(fc_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = x.transpose(1, 2)  # from (batch, features, time) to (batch, time, features)
        
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        
        # Use the final time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    """LSTM model for speech command classification"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer size depends on bidirectional setting
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layer
        self.fc = nn.Linear(fc_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = x.transpose(1, 2)  # from (batch, features, time) to (batch, time, features)
        
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the final time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Classification
        out = self.fc(out)
        return out
    

class ConformerBlock(nn.Module):
    """
    Conformer block combining self-attention and convolution for speech processing.
    """
    def __init__(self, dim, num_heads, kernel_size=31, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        # Feed Forward Module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-Head Self-Attention
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_attention = nn.Dropout(dropout)
        
        # Convolution Module
        self.norm_conv = nn.LayerNorm(dim)
        self.conv_module = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),  # Pointwise conv
            nn.GLU(dim=1),  # Gated Linear Unit
            # Depthwise conv
            nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                      padding=(kernel_size-1)//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),  # Swish activation
            nn.Conv1d(dim, dim, kernel_size=1),  # Pointwise conv
            nn.Dropout(dropout)
        )
        
        # Feed Forward Module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Feed Forward Module 1 with residual connection
        x = x + 0.5 * self.ff1(x)
        
        # Multi-Head Self-Attention with residual connection
        attn_x = self.norm_attention(x)
        attn_x, _ = self.attention(attn_x, attn_x, attn_x)
        x = x + self.dropout_attention(attn_x)
        
        # Convolution Module with residual connection
        conv_x = self.norm_conv(x)
        batch, seq_len, dim = conv_x.shape
        conv_x = conv_x.transpose(1, 2)  # (B, D, T)
        conv_x = self.conv_module(conv_x)
        conv_x = conv_x.transpose(1, 2)  # (B, T, D)
        x = x + conv_x
        
        # Feed Forward Module 2 with residual connection
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class ConformerClassifier(nn.Module):
    """
    Conformer-based classifier for speech command recognition.
    Combines convolution and self-attention mechanisms.
    """
    def __init__(self, 
                n_classes,
                n_mfcc=40,
                d_model=144, 
                num_layers=4,
                num_heads=4,
                kernel_size=31,
                expansion_factor=4,
                dropout=0.1,
                pool="mean"):
        super().__init__()
        self.pool = pool
        
        # Input projection and positional encoding
        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        # CLS token for classification if needed
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Stack of Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim=d_model,
                num_heads=num_heads,
                kernel_size=kernel_size,
                expansion_factor=expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):  # x: (B, 40, 101)
        x = x.transpose(1, 2)  # (B, 101, 40)
        x = self.input_proj(x)  # (B, 101, d_model)
        
        # Add CLS token if using cls pooling
        if self.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)  # (B, 102, d_model)
            
        x = self.pos_enc(x)
        
        # Process through Conformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Pooling: either use CLS token or mean pooling
        if self.pool == "cls":
            rep = x[:, 0]  # Take CLS token representation
        else:
            rep = x.mean(dim=1)  # Mean pooling
            
        return self.classifier(rep)
    

class DualPathTransformer(nn.Module):
    """
    Dual-Path Transformer (DPT) for speech command recognition.
    
    Processes the input through two parallel paths:
    1. Frequency path: operates along the time axis to capture temporal patterns
    2. Temporal path: operates along the frequency axis to capture spectral patterns
    
    The outputs from both paths are then fused for final classification.
    """
    def __init__(self, 
                n_classes,
                n_mfcc=40,
                d_model=128, 
                n_heads=4,
                num_layers=4, 
                dim_ff=256,
                dropout=0.1,
                fusion_method="concat"):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Input projection
        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        # Frequency path (operates along time axis)
        freq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.freq_encoder = nn.TransformerEncoder(
            freq_encoder_layer, num_layers=num_layers//2)
        
        # Temporal path (operates along frequency axis)
        # We'll use a different approach for the temporal path that
        # correctly maintains dimensions
        temp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.temp_encoder = nn.TransformerEncoder(
            temp_encoder_layer, num_layers=num_layers//2)
        
        # Cross-path fusion
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_encoder_layer, num_layers=num_layers//2)
        
        # Alternative fusion method
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
        # Final classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):  # x: (B, 40, 101)
        batch_size = x.size(0)
        
        # Frequency path - process along time axis
        freq_x = x.transpose(1, 2)  # (B, 101, 40)
        freq_x = self.input_proj(freq_x)  # (B, 101, d_model)
        freq_x = self.pos_enc(freq_x)
        freq_out = self.freq_encoder(freq_x)  # (B, 101, d_model)
        
        # Temporal path - process along frequency axis
        # Instead of reshaping in a way that loses dimension info,
        # we'll transpose x and process it directly
        temp_x = x.permute(0, 2, 1)  # (B, 101, 40)
        temp_x = self.input_proj(temp_x)  # (B, 101, d_model)
        temp_x = self.pos_enc(temp_x)
        temp_out = self.temp_encoder(temp_x)  # (B, 101, d_model)
        
        # Fusion of both paths (both have shape B, 101, d_model)
        if self.fusion_method == "add":
            fusion = freq_out + temp_out
        elif self.fusion_method == "multiply":
            fusion = freq_out * temp_out
        elif self.fusion_method == "concat":
            fusion = torch.cat([freq_out, temp_out], dim=2)  # (B, 101, d_model*2)
            fusion = self.fusion_proj(fusion)  # (B, 101, d_model)
        else:  # Default to add
            fusion = freq_out + temp_out
            
        # Process through fusion encoder
        fusion = self.fusion_encoder(fusion)
        
        # Global average pooling
        rep = fusion.mean(dim=1)  # (B, d_model)
        
        # Classification
        return self.classifier(rep)


class DualPathTransformerV2(nn.Module):
    """
    Improved Dual-Path Transformer for speech command recognition.
    
    This version uses a more direct approach for the temporal path by
    simply transposing the input tensor and processing it with a transformer.
    """
    def __init__(self, 
                n_classes,
                n_mfcc=40,
                d_model=128, 
                n_heads=4,
                num_layers=4, 
                dim_ff=256,
                dropout=0.1,
                fusion_method="concat"):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Input projections for both paths
        self.freq_proj = nn.Linear(n_mfcc, d_model)
        self.temp_proj = nn.Linear(101, d_model)  # assuming 101 time steps
        
        self.freq_pos_enc = PositionalEncoding(d_model)
        self.temp_pos_enc = PositionalEncoding(d_model)
        
        # Frequency path (operates along time axis)
        freq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.freq_encoder = nn.TransformerEncoder(
            freq_encoder_layer, num_layers=num_layers//2)
        
        # Temporal path (operates along frequency axis)
        temp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.temp_encoder = nn.TransformerEncoder(
            temp_encoder_layer, num_layers=num_layers//2)
        
        # Cross-path fusion
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_encoder_layer, num_layers=num_layers//2)
        
        # Alternative fusion methods
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
        # Final classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):  # x: (B, 40, 101)
        batch_size = x.size(0)
        
        # Frequency path - process along time axis
        freq_x = x.transpose(1, 2)  # (B, 101, 40)
        freq_x = self.freq_proj(freq_x)  # (B, 101, d_model)
        freq_x = self.freq_pos_enc(freq_x)
        freq_out = self.freq_encoder(freq_x)  # (B, 101, d_model)
        
        # Temporal path - process along frequency axis
        temp_x = x  # (B, 40, 101)
        temp_x = self.temp_proj(temp_x)  # (B, 40, d_model)
        temp_x = self.temp_pos_enc(temp_x)
        temp_out = self.temp_encoder(temp_x)  # (B, 40, d_model)
        
        # Align dimensions for fusion (both should be B, seq_len, d_model)
        # Use average pooling to make dimensions compatible
        temp_out = torch.mean(temp_out, dim=1, keepdim=True)  # (B, 1, d_model)
        temp_out = temp_out.expand(-1, freq_out.size(1), -1)  # (B, 101, d_model)
        
        # Fusion of both paths
        if self.fusion_method == "add":
            fusion = freq_out + temp_out
        elif self.fusion_method == "multiply":
            fusion = freq_out * temp_out
        elif self.fusion_method == "concat":
            fusion = torch.cat([freq_out, temp_out], dim=2)  # (B, 101, d_model*2)
            fusion = self.fusion_proj(fusion)  # (B, 101, d_model)
        elif self.fusion_method == "gated":
            concat = torch.cat([freq_out, temp_out], dim=2)  # (B, 101, d_model*2)
            gate = self.gate(concat)
            fusion = gate * freq_out + (1 - gate) * temp_out
        else:  # Default to add
            fusion = freq_out + temp_out
            
        # Process through fusion encoder
        fusion = self.fusion_encoder(fusion)
        
        # Global average pooling
        rep = fusion.mean(dim=1)  # (B, d_model)
        
        # Classification
        return self.classifier(rep)
    

class HierarchicalTransformerFP(nn.Module):
    """
    Hierarchical Transformer with Feature Pyramid (HT-FP) for speech command recognition.
    
    Processes the input at multiple resolutions, capturing both fine-grained
    and coarse patterns in the MFCC features.
    """
    def __init__(self, 
                n_classes,
                n_mfcc=40,
                d_model=128, 
                n_heads=4,
                num_layers=4, 
                dim_ff=256,
                dropout=0.1,
                fusion_method="concat"):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Feature pyramid levels projections
        self.level1_proj = nn.Linear(n_mfcc, d_model)
        self.level2_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.level2_proj = nn.Linear(n_mfcc, d_model)
        self.level3_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.level3_proj = nn.Linear(n_mfcc, d_model)
        
        # Positional encoding for all levels
        self.pos_enc = PositionalEncoding(d_model)
        
        # Level 1 transformer (highest resolution)
        l1_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l1_encoder = nn.TransformerEncoder(
            l1_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Level 2 transformer (medium resolution)
        l2_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l2_encoder = nn.TransformerEncoder(
            l2_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Level 3 transformer (lowest resolution)
        l3_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l3_encoder = nn.TransformerEncoder(
            l3_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Fusion projections
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(d_model * 3, d_model)
        elif fusion_method == "weighted":
            self.level_weights = nn.Parameter(torch.ones(3) / 3)
            self.softmax = nn.Softmax(dim=0)
        elif fusion_method == "gated":
            self.gate1 = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            self.gate2 = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            self.gate3 = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
        
        # Final classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):  # x: (B, 40, 101)
        batch_size = x.size(0)
        
        # Level 1 - full resolution
        x1 = x.transpose(1, 2)  # (B, 101, 40)
        x1 = self.level1_proj(x1)  # (B, 101, d_model)
        x1 = self.pos_enc(x1)
        x1 = self.l1_encoder(x1)
        x1_rep = x1.mean(dim=1)  # (B, d_model)
        
        # Level 2 - halved resolution
        x2 = self.level2_pool(x)  # (B, 40, 50)
        x2 = x2.transpose(1, 2)  # (B, 50, 40)
        x2 = self.level2_proj(x2)  # (B, 50, d_model)
        x2 = self.pos_enc(x2)
        x2 = self.l2_encoder(x2)
        x2_rep = x2.mean(dim=1)  # (B, d_model)
        
        # Level 3 - quarter resolution
        x3 = self.level3_pool(x)  # (B, 40, 25)
        x3 = x3.transpose(1, 2)  # (B, 25, 40)
        x3 = self.level3_proj(x3)  # (B, 25, d_model)
        x3 = self.pos_enc(x3)
        x3 = self.l3_encoder(x3)
        x3_rep = x3.mean(dim=1)  # (B, d_model)
        
        # Fusion of representations from all levels
        if self.fusion_method == "add":
            rep = x1_rep + x2_rep + x3_rep
        elif self.fusion_method == "concat":
            rep = torch.cat([x1_rep, x2_rep, x3_rep], dim=1)  # (B, d_model*3)
            rep = self.fusion_proj(rep)  # (B, d_model)
        elif self.fusion_method == "weighted":
            weights = self.softmax(self.level_weights)
            rep = weights[0] * x1_rep + weights[1] * x2_rep + weights[2] * x3_rep
        elif self.fusion_method == "gated":
            gate1 = self.gate1(x1_rep)  # (B, 1)
            gate2 = self.gate2(x2_rep)  # (B, 1)
            gate3 = self.gate3(x3_rep)  # (B, 1)
            
            # Normalize gates to sum to 1
            gates_sum = gate1 + gate2 + gate3
            gate1 = gate1 / gates_sum
            gate2 = gate2 / gates_sum
            gate3 = gate3 / gates_sum
            
            rep = gate1 * x1_rep + gate2 * x2_rep + gate3 * x3_rep
        else:  # Default to average
            rep = (x1_rep + x2_rep + x3_rep) / 3
        
        # Classification
        return self.classifier(rep)


class HierarchicalTransformerFPV2(nn.Module):
    """
    Enhanced version of Hierarchical Transformer with Feature Pyramid.
    
    Key improvements:
    1. Cross-attention between different levels
    2. Pyramid feature exchange
    3. Enhanced fusion techniques
    """
    def __init__(self, 
                n_classes,
                n_mfcc=40,
                d_model=128, 
                n_heads=4,
                num_layers=4, 
                dim_ff=256,
                dropout=0.1,
                cross_level_attn=True,  # Enable cross-level attention
                fusion_method="concat"):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.cross_level_attn = cross_level_attn
        
        # Feature pyramid levels projections
        self.level1_proj = nn.Linear(n_mfcc, d_model)
        self.level2_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.level2_proj = nn.Linear(n_mfcc, d_model)
        self.level3_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.level3_proj = nn.Linear(n_mfcc, d_model)
        
        # Positional encoding for all levels
        self.pos_enc = PositionalEncoding(d_model)
        
        # Level 1 transformer (highest resolution)
        l1_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l1_encoder = nn.TransformerEncoder(
            l1_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Level 2 transformer (medium resolution)
        l2_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l2_encoder = nn.TransformerEncoder(
            l2_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Level 3 transformer (lowest resolution)
        l3_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True)
        self.l3_encoder = nn.TransformerEncoder(
            l3_encoder_layer, num_layers=num_layers//3 if num_layers >= 3 else 1)
        
        # Cross-level attention
        if cross_level_attn:
            # L1 -> L2 attention and L1 -> L3 attention
            self.l1_to_l2_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.l1_to_l3_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            
            # L2 -> L1 attention and L2 -> L3 attention
            self.l2_to_l1_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.l2_to_l3_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            
            # L3 -> L1 attention and L3 -> L2 attention
            self.l3_to_l1_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.l3_to_l2_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            
            # Layer norms for cross-attention
            self.norm_l1 = nn.LayerNorm(d_model)
            self.norm_l2 = nn.LayerNorm(d_model)
            self.norm_l3 = nn.LayerNorm(d_model)
        
        # Fusion projections
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(d_model * 3, d_model)
        elif fusion_method == "weighted":
            self.level_weights = nn.Parameter(torch.ones(3) / 3)
            self.softmax = nn.Softmax(dim=0)
        elif fusion_method == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model * 3),
                nn.Sigmoid()
            )
        elif fusion_method == "hierarchical":
            # Hierarchical fusion network
            self.fusion_l1_l2 = nn.Linear(d_model * 2, d_model)
            self.fusion_l1l2_l3 = nn.Linear(d_model * 2, d_model)
        
        # Final classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):  # x: (B, 40, 101)
        batch_size = x.size(0)
        
        # Level 1 - full resolution
        x1 = x.transpose(1, 2)  # (B, 101, 40)
        x1 = self.level1_proj(x1)  # (B, 101, d_model)
        x1 = self.pos_enc(x1)
        x1 = self.l1_encoder(x1)  # (B, 101, d_model)
        
        # Level 2 - halved resolution
        x2 = self.level2_pool(x)  # (B, 40, 50)
        x2 = x2.transpose(1, 2)  # (B, 50, 40)
        x2 = self.level2_proj(x2)  # (B, 50, d_model)
        x2 = self.pos_enc(x2)
        x2 = self.l2_encoder(x2)  # (B, 50, d_model)
        
        # Level 3 - quarter resolution
        x3 = self.level3_pool(x)  # (B, 40, 25)
        x3 = x3.transpose(1, 2)  # (B, 25, 40)
        x3 = self.level3_proj(x3)  # (B, 25, d_model)
        x3 = self.pos_enc(x3)
        x3 = self.l3_encoder(x3)  # (B, 25, d_model)
        
        # Apply cross-level attention if enabled
        if self.cross_level_attn:
            # Interpolate x2 to match x1 length for cross-attention
            x2_upsampled = torch.nn.functional.interpolate(
                x2.transpose(1, 2), size=x1.size(1), mode='linear'
            ).transpose(1, 2)
            
            # Interpolate x3 to match x1 and x2 lengths
            x3_upsampled_l1 = torch.nn.functional.interpolate(
                x3.transpose(1, 2), size=x1.size(1), mode='linear'
            ).transpose(1, 2)
            
            x3_upsampled_l2 = torch.nn.functional.interpolate(
                x3.transpose(1, 2), size=x2.size(1), mode='linear'
            ).transpose(1, 2)
            
            # Apply cross-attention between levels
            # L1 with L2 and L3
            x1_l2_attn, _ = self.l1_to_l2_attn(
                self.norm_l1(x1), x2_upsampled, x2_upsampled
            )
            x1_l3_attn, _ = self.l1_to_l3_attn(
                self.norm_l1(x1), x3_upsampled_l1, x3_upsampled_l1
            )
            
            # L2 with L1 and L3
            x2_l1_attn, _ = self.l2_to_l1_attn(
                self.norm_l2(x2), x1[:, :x2.size(1)], x1[:, :x2.size(1)]
            )
            x2_l3_attn, _ = self.l2_to_l3_attn(
                self.norm_l2(x2), x3_upsampled_l2, x3_upsampled_l2
            )
            
            # L3 with L1 and L2
            x3_l1_attn, _ = self.l3_to_l1_attn(
                self.norm_l3(x3), x1[:, :x3.size(1)], x1[:, :x3.size(1)]
            )
            x3_l2_attn, _ = self.l3_to_l2_attn(
                self.norm_l3(x3), x2[:, :x3.size(1)], x2[:, :x3.size(1)]
            )
            
            # Add residual connections
            x1 = x1 + x1_l2_attn + x1_l3_attn
            x2 = x2 + x2_l1_attn + x2_l3_attn
            x3 = x3 + x3_l1_attn + x3_l2_attn
        
        # Global pooling on each level
        x1_rep = x1.mean(dim=1)  # (B, d_model)
        x2_rep = x2.mean(dim=1)  # (B, d_model)
        x3_rep = x3.mean(dim=1)  # (B, d_model)
        
        # Fusion of representations from all levels
        if self.fusion_method == "add":
            rep = x1_rep + x2_rep + x3_rep
        elif self.fusion_method == "concat":
            rep = torch.cat([x1_rep, x2_rep, x3_rep], dim=1)  # (B, d_model*3)
            rep = self.fusion_proj(rep)  # (B, d_model)
        elif self.fusion_method == "weighted":
            weights = self.softmax(self.level_weights)
            rep = weights[0] * x1_rep + weights[1] * x2_rep + weights[2] * x3_rep
        elif self.fusion_method == "gated":
            concat_rep = torch.cat([x1_rep, x2_rep, x3_rep], dim=1)  # (B, d_model*3)
            gates = self.fusion_gate(concat_rep)  # (B, d_model*3)
            
            # Split gates for each level
            g1, g2, g3 = torch.chunk(gates, 3, dim=1)  # Each (B, d_model)
            
            # Apply gates and sum
            rep = g1 * x1_rep + g2 * x2_rep + g3 * x3_rep
        elif self.fusion_method == "hierarchical":
            # First fuse level 1 and 2
            l1_l2 = torch.cat([x1_rep, x2_rep], dim=1)
            l1_l2_fused = self.fusion_l1_l2(l1_l2)
            
            # Then fuse result with level 3
            l1l2_l3 = torch.cat([l1_l2_fused, x3_rep], dim=1)
            rep = self.fusion_l1l2_l3(l1l2_l3)
        else:  # Default to average
            rep = (x1_rep + x2_rep + x3_rep) / 3
        
        # Classification
        return self.classifier(rep)