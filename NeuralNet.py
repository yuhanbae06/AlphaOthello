import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

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
        
# PoolFormer

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim) 
        
        self.token_mixer = nn.AvgPool2d(
            kernel_size=pool_size, 
            stride=1, 
            padding=pool_size//2, 
            count_include_pad=False
        )
        
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1), # Channel Expansion
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)  # Channel Reduction
        )
        
        # 스케일 보정 (학습 안정성)
        self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones((dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-5 * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        # x: (B, C, H, W)
        
        # 1. Token Mixing (Pooling)
        input = x
        x = self.norm1(x)
        x = self.token_mixer(x) - x
        x = input + self.layer_scale_1 * x
        
        # 2. Channel Mixing (MLP)
        input = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = input + self.layer_scale_2 * x
        
        return x

class AlphaZeroPoolFormer(nn.Module):
    def __init__(self, game, num_blocks, dim, device):
        super().__init__()
        self.device = device
        
        # Patch Embedding
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList(
            [PoolFormerBlock(dim) for _ in range(num_blocks)]
        )
        
        # Heads
        self.policyHead = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Flatten(),
            nn.Linear(dim * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Flatten(),
            nn.Linear(dim * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.policyHead(x), self.valueHead(x)