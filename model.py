"""
SSD Trace 雙 Head 神經網路模型架構 (model.py)
重構重點:
1. 混合重建損失: 3 個連續特徵採用 NMSE Loss, IsWrite 採 BCEWithLogitsLoss
2. 保持 1D Conv + SE Attention + FC Bottleneck 簡潔結構
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_nmse(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """計算連續特徵的 NMSE"""
    mse = torch.sum((x_true - x_pred) ** 2, dim=(1, 2))
    var = torch.sum(x_true ** 2, dim=(1, 2)) + eps
    return torch.mean(mse / var)


def compute_hybrid_recon_loss(x_true: torch.Tensor, x_recon_cont: torch.Tensor, logits_iswrite: torch.Tensor, bce_weight: float = 0.5):
    """
    【修正點 1】混合重建損失:
    - Continuous (前 3 通道: DeltaLBA, DeltaTime, SizeKB): NMSE Loss
    - Binary (第 4 通道: IsWrite): BCEWithLogitsLoss
    """
    x_true_cont = x_true[:, :, :3]
    x_true_write = x_true[:, :, 3]  # (B, T)
    
    loss_nmse = compute_nmse(x_true_cont, x_recon_cont)
    loss_bce = F.binary_cross_entropy_with_logits(logits_iswrite.squeeze(-1), x_true_write)
    
    total_recon_loss = loss_nmse + bce_weight * loss_bce
    return total_recon_loss, loss_nmse.item(), loss_bce.item()


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        weights = self.fc(x).view(b, c, 1)
        return x * weights


class SharedEncoder(nn.Module):
    def __init__(self, in_features: int = 4, seq_len: int = 32, latent_dim: int = 16):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock1D(channels=32),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SEBlock1D(channels=64),
            
            nn.MaxPool1d(kernel_size=2)  # (B, 64, 16)
        )
        
        flatten_dim = 64 * (seq_len // 2)  # 1024
        self.fc_bottleneck = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        x = self.conv_block(x)
        x = x.flatten(start_dim=1)
        z = self.fc_bottleneck(x)
        return z


class ReconstructionHead(nn.Module):
    def __init__(self, latent_dim: int = 16, seq_len: int = 32):
        super().__init__()
        self.seq_len = seq_len
        
        # 解碼器輸出 3 個連續欄位 + 1 個 IsWrite Logit
        self.decoder_cont = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * 3)
        )
        self.decoder_write = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len * 1)
        )

    def forward(self, z: torch.Tensor):
        x_cont = self.decoder_cont(z).view(-1, self.seq_len, 3)
        logits_write = self.decoder_write(z).view(-1, self.seq_len, 1)
        return x_cont, logits_write


class HotColdHead(nn.Module):
    def __init__(self, latent_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class DualHeadTraceModel(nn.Module):
    def __init__(self, in_features: int = 4, seq_len: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = SharedEncoder(in_features=in_features, seq_len=seq_len, latent_dim=latent_dim)
        self.head_a = ReconstructionHead(latent_dim=latent_dim, seq_len=seq_len)
        self.head_b = HotColdHead(latent_dim=latent_dim, num_classes=2)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon_cont, logits_iswrite = self.head_a(z)
        logits_hot = self.head_b(z)
        return x_recon_cont, logits_iswrite, logits_hot, z