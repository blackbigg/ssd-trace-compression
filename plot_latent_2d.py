"""
2D Latent 空間特徵聚類與決策邊界視覺化 (plot_latent_2d.py)

目的：
針對 z=2 (64 倍極限壓縮) 的最佳模型，抽樣 Test Set 視窗，
將 Latent 向量 z = [z1, z2] 直接繪製於 2D 平面，觀察 Hot / Cold 的幾何聚類效果。
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import DualHeadTraceModel

PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")
CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")
CKPT_PATH_Z2 = CHECKPOINT_DIR / "best_model_z2.pt"
OUTPUT_IMAGE = CHECKPOINT_DIR / "latent_space_2d.png"

# 設定樣式與中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

SAMPLE_SIZE = 6000  # 抽樣 6000 個測試視窗以利清晰繪圖，避免點過於密集的繪圖遮蓋


def extract_latent_vectors():
    print("1. 載入測試集數據與 z=2 最佳模型...")
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_test, y_test = data["test_X"], data["test_y"]

    # 隨機固定種子抽樣 SAMPLE_SIZE 個視窗
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices].numpy()

    device = torch.device("cpu")
    model = DualHeadTraceModel(in_features=4, seq_len=32, latent_dim=2)
    model.load_state_dict(torch.load(CKPT_PATH_Z2, map_location=device))
    model.eval()

    print("2. 導出 Latent 向量 (z1, z2)...")
    with torch.no_grad():
        _, _, logits_hot, z_sample = model(X_sample)
        preds = torch.argmax(logits_hot, dim=1).numpy()

    z_sample = z_sample.numpy()
    return z_sample, y_sample, preds


def plot_latent_2d():
    z_sample, y_true, y_pred = extract_latent_vectors()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 區分 Cold (0) 與 Hot (1)
    cold_mask = (y_true == 0)
    hot_mask = (y_true == 1)

    # 繪製散佈點 (Alpha 設為 0.4 避免 Overplotting)
    scatter_cold = ax.scatter(
        z_sample[cold_mask, 0], z_sample[cold_mask, 1],
        c='#1f77b4', label=f'Cold Windows (n={cold_mask.sum():,})',
        alpha=0.4, s=15, edgecolors='none'
    )
    scatter_hot = ax.scatter(
        z_sample[hot_mask, 0], z_sample[hot_mask, 1],
        c='#d62728', label=f'Hot Windows (n={hot_mask.sum():,})',
        alpha=0.5, s=15, edgecolors='none'
    )

    ax.set_xlabel('Latent Dimension 1 ($z_1$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latent Dimension 2 ($z_2$)', fontsize=11, fontweight='bold')
    ax.set_title('2D Latent Representation Space ($z=2$, 64x Compression)', fontsize=12, fontweight='bold', pad=12)

    # 加強圖例與外框
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.close()
    print(f"✅ 2D Latent 聚類圖表已成功生成: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    plot_latent_2d()