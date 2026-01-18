# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: AlphaOthello
#     language: python
#     name: alphaothello
# ---

# +
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# +
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_hidden = num_hidden

        self.norm1 = nn.LayerNorm(num_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=num_hidden, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(num_hidden)
        self.ff = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.GELU(),
            nn.Linear(num_hidden * 4, num_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        seq = x.view(b, c, h * w).permute(2, 0, 1)

        seq2 = self.norm1(seq)
        attn_out, _ = self.attn(seq2, seq2, seq2)
        seq = seq + attn_out

        seq2 = self.norm2(seq)
        seq2 = self.ff(seq2)
        seq = seq + seq2

        out = seq.permute(1, 2, 0).view(b, c, h, w)
        return out
        
class ResTNet(nn.Module):
    def __init__(self, game, num_hidden, device, block_types='RRTRRT', num_heads=4, dropout=0.1):
        super().__init__()

        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        blocks = []
        for b in block_types:
            if b.upper() == 'R':
                blocks.append(ResBlock(num_hidden))
            elif b.upper() == 'T':
                blocks.append(TransformerBlock(num_hidden, num_heads, dropout))
            else:
                raise ValueError(f"Unknown block type '{b}' (expected 'R' or 'T')")
        self.backBone = nn.ModuleList(blocks)

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for blk in self.backBone:
            x = blk(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value