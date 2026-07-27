"""
修正版：帕累托前沿與 Rate-Distortion 雙軸繪圖腳本 (plot_pareto.py)

修正重點：
1. 改用等距類別軸 (x_positions)，讓 z = 2, 4, 8, 16, 24, 32, 64 均勻排列。
2. 徹底解決 z=2 與 z=4 標籤重疊擁擠的問題。
3. 適度拉寬 Y 軸範圍與 Padding，提升視覺質感。
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")
JSON_PATH = CHECKPOINT_DIR / "sweep_results.json"
OUTPUT_IMAGE = CHECKPOINT_DIR / "pareto_frontier.png"

# 設定字型與樣式
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_pareto_curve():
    if not JSON_PATH.exists():
        print(f"找不到檔案: {JSON_PATH}，請確認訓練已完成。")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 依 Latent Dim 排序
    data = sorted(data, key=lambda x: x["latent_dim"])

    dims = [item["latent_dim"] for item in data]
    f1_scores = [item["test_f1"] for item in data]
    auc_scores = [item["test_auc"] for item in data]
    nmse_conts = [item["test_nmse_cont"] for item in data]
    bce_writes = [item["test_bce_write"] for item in data]

    # 【核心修正】：改用 index 索引 [0, 1, 2, 3, 4, 5, 6] 作為等距 X 軸位置
    x_positions = list(range(len(dims)))

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    # --- 左 Y 軸: 分類指標 (F1 & AUC) ---
    color_f1 = '#1f77b4'   # 藍色
    color_auc = '#2ca02c'  # 綠色
    
    line1 = ax1.plot(x_positions, f1_scores, color=color_f1, marker='o', linewidth=2.5, markersize=7, label='Test F1-Score (Classification)')
    line2 = ax1.plot(x_positions, auc_scores, color=color_auc, marker='s', linestyle='--', linewidth=2.0, markersize=6, label='Test ROC-AUC')
    
    ax1.set_xlabel('Latent Bottleneck Dimension (z)', fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Classification Performance (F1 / AUC)', fontsize=11, fontweight='bold', color=color_f1)
    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.set_ylim(0.88, 1.005)  # 微調 Y 軸範圍，更凸顯高水平平坦趨勢

    # 設定等距類別刻度與標籤
    ax1.set_xticks(x_positions)
    tick_labels = [f"{d}\n({128/d:.0f}x)" for d in dims]
    ax1.set_xticklabels(tick_labels, fontsize=9.5)

    # --- 右 Y 軸: 重建指標 (NMSE & BCE) ---
    ax2 = ax1.twinx()
    color_nmse = '#d62728'  # 紅色
    color_bce = '#ff7f0e'   # 橘色
    
    line3 = ax2.plot(x_positions, nmse_conts, color=color_nmse, marker='^', linewidth=2.0, markersize=7, label='Continuous NMSE (Recon Loss)')
    line4 = ax2.plot(x_positions, bce_writes, color=color_bce, marker='d', linestyle=':', linewidth=1.5, markersize=6, label='IsWrite BCE (Recon Loss)')
    
    ax2.set_ylabel('Reconstruction Loss (NMSE / BCE)', fontsize=11, fontweight='bold', color=color_nmse)
    ax2.tick_params(axis='y', labelcolor=color_nmse)
    ax2.set_ylim(-0.02, max(nmse_conts) * 1.15)
    ax2.grid(False)

    # 組合圖例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.title('Latent Dimension vs. Downstream Performance & Reconstruction Loss', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.close()
    print(f"✅ 修正版帕累托前沿圖表已成功重繪: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_pareto_curve()