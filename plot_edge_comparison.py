"""
邊緣可行性綜合比較圖繪製腳本 (plot_edge_comparison.py)

繪製 z in {2, 8, 64} 在 FP32 與 INT8 條件下之：
1. 模型體積 (MB)
2. 分類 F1-Score
3. CPU 單筆推論延遲 (ms/window)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")
OUTPUT_IMAGE = CHECKPOINT_DIR / "edge_comparison.png"

# 設定字型與視覺樣式
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_edge_comparison():
    dims = ['z = 2\n(64x)', 'z = 8\n(16x)', 'z = 64\n(2x)']
    x = np.arange(len(dims))
    width = 0.35

    # 數據來源 (Section 5.4 / Section 8)
    fp32_sizes = [0.595, 0.603, 0.679]
    int8_sizes = [0.191, 0.193, 0.213]

    fp32_f1s = [0.9171, 0.9163, 0.9138]
    int8_f1s = [0.9168, 0.9156, 0.9140]

    fp32_lats = [0.641, 0.919, 0.766]
    int8_lats = [2.152, 2.309, 2.172]

    color_fp32 = '#1f77b4'  # 藍色
    color_int8 = '#ff7f0e'  # 橘色

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # --- 1. 模型體積 (Model Size) ---
    rects1 = ax1.bar(x - width/2, fp32_sizes, width, label='FP32', color=color_fp32)
    rects2 = ax1.bar(x + width/2, int8_sizes, width, label='INT8 (Quantized)', color=color_int8)
    ax1.set_ylabel('Model Size (MB)', fontsize=10, fontweight='bold')
    ax1.set_title('Model Size Footprint (MB)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims, fontsize=9.5)
    ax1.set_ylim(0, 0.85)
    ax1.bar_label(rects1, fmt='%.3f', padding=3, fontsize=8)
    ax1.bar_label(rects2, fmt='%.3f', padding=3, fontsize=8)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)

    # --- 2. 分類 F1 分數 (Test F1 Score) ---
    rects3 = ax2.bar(x - width/2, fp32_f1s, width, label='FP32', color=color_fp32)
    rects4 = ax2.bar(x + width/2, int8_f1s, width, label='INT8 (Quantized)', color=color_int8)
    ax2.set_ylabel('Test F1-Score', fontsize=10, fontweight='bold')
    ax2.set_title('Classification Performance (F1)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dims, fontsize=9.5)
    ax2.set_ylim(0.90, 0.93)  # 放大微小差異
    ax2.bar_label(rects3, fmt='%.4f', padding=3, fontsize=8)
    ax2.bar_label(rects4, fmt='%.4f', padding=3, fontsize=8)
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)

    # --- 3. CPU 推論延遲 (Inference Latency, bs=1) ---
    rects5 = ax3.bar(x - width/2, fp32_lats, width, label='FP32', color=color_fp32)
    rects6 = ax3.bar(x + width/2, int8_lats, width, label='INT8 (Quantized)', color=color_int8)
    ax3.set_ylabel('Latency (ms / window)', fontsize=10, fontweight='bold')
    ax3.set_title('CPU Inference Latency (batch=1)', fontsize=11, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dims, fontsize=9.5)
    ax3.set_ylim(0, 2.8)
    ax3.bar_label(rects5, fmt='%.3f', padding=3, fontsize=8)
    ax3.bar_label(rects6, fmt='%.3f', padding=3, fontsize=8)
    ax3.legend(loc='upper left', frameon=True, fontsize=8.5)

    plt.suptitle('Edge Feasibility Benchmark across Latent Dimensions (z = 2, 8, 64)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 邊緣可行性綜合比較圖已成功生成: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_edge_comparison()