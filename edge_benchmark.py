"""
邊緣可行性驗證 (edge_benchmark.py)

針對已經訓練好的幾個代表性 latent 維度，量測：
1. 模型參數量與檔案大小 (fp32)
2. CPU-only 推論延遲（單筆 batch=1，模擬邊緣裝置即時推論；batch=64 模擬批次處理）
3. 動態量化 (Dynamic Quantization) 後的模型大小、延遲、F1 精度變化

限制說明：PyTorch 的 dynamic quantization 目前只支援 nn.Linear / nn.LSTM 等層，
Conv1d 不在支援範圍內，因此量化只作用在 FC bottleneck 與兩個 Head 的線性層。
若要對 Conv1d 也做 int8 量化，需要用 static quantization（需要校準資料集），
以你目前的時程先用 dynamic quantization 拿到一個保守但可信的量化效果估計即可。

使用方式：
    python edge_benchmark.py
需要 model.py（DualHeadTraceModel）與已訓練好的 checkpoints 在同一路徑下。
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import DualHeadTraceModel


PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")
CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")

# 挑幾個代表性維度：2（極致壓縮）、8（甜蜜點）、64（原始規劃上限）
DIMS_TO_BENCHMARK = [2, 8, 64]

SEQ_LEN = 32
IN_FEATURES = 4
N_WARMUP = 20
N_TIMED_RUNS = 200
BATCH_SIZES = [1, 64]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """粗略估計 fp32 模型大小 (MB)，每個參數 4 bytes"""
    return count_params(model) * 4 / (1024 ** 2)


@torch.no_grad()
def benchmark_latency(model: nn.Module, batch_size: int, device: torch.device) -> float:
    """回傳平均每筆 (ms/window) 的推論延遲"""
    model.eval()
    dummy = torch.randn(batch_size, SEQ_LEN, IN_FEATURES, device=device)

    # Warm-up，排除第一次呼叫的初始化開銷
    for _ in range(N_WARMUP):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_TIMED_RUNS):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_windows = N_TIMED_RUNS * batch_size
    ms_per_window = (elapsed / total_windows) * 1000
    return ms_per_window


@torch.no_grad()
def evaluate_f1(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds, all_targets = [], []
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        _, _, logits_hot, _ = model(batch_X)
        preds = torch.argmax(logits_hot, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.numpy())
    return f1_score(all_targets, all_preds, zero_division=0)


def load_test_loader():
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_test, y_test = data["test_X"], data["test_y"]
    return DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)


def benchmark_one_dim(latent_dim: int, test_loader: DataLoader, cpu_device: torch.device):
    print(f"\n{'='*70}")
    print(f" Latent Dim = {latent_dim} 邊緣可行性評測")
    print(f"{'='*70}")

    ckpt_path = CHECKPOINT_DIR / f"best_model_z{latent_dim}.pt"
    model_fp32 = DualHeadTraceModel(in_features=IN_FEATURES, seq_len=SEQ_LEN, latent_dim=latent_dim)
    model_fp32.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_fp32.to(cpu_device)

    # --- FP32 baseline ---
    n_params = count_params(model_fp32)
    size_fp32 = model_size_mb(model_fp32)
    f1_fp32 = evaluate_f1(model_fp32, test_loader, cpu_device)

    print(f"參數量: {n_params:,}")
    print(f"FP32 模型大小: {size_fp32:.3f} MB")
    print(f"FP32 Test F1: {f1_fp32:.4f}")

    latency_fp32 = {}
    for bs in BATCH_SIZES:
        ms = benchmark_latency(model_fp32, bs, cpu_device)
        latency_fp32[bs] = ms
        print(f"FP32 CPU 推論延遲 (batch={bs:3d}): {ms:.4f} ms/window")

    # --- 動態量化 (僅作用於 Linear 層) ---
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    size_int8 = sum(
        p.numel() * (1 if hasattr(p, "dtype") and p.dtype == torch.qint8 else 4)
        for p in model_int8.parameters()
    ) / (1024 ** 2)
    # 上面的估計不夠精確（量化參數不是標準 nn.Parameter），改用實際存檔大小更準確
    tmp_path = Path("_tmp_quantized_model.pt")
    torch.save(model_int8.state_dict(), tmp_path)
    size_int8_actual = tmp_path.stat().st_size / (1024 ** 2)
    tmp_path.unlink()

    f1_int8 = evaluate_f1(model_int8, test_loader, cpu_device)

    print(f"\nINT8 動態量化後模型大小: {size_int8_actual:.3f} MB (壓縮 {size_fp32/size_int8_actual:.2f}x)")
    print(f"INT8 Test F1: {f1_int8:.4f} (相對 FP32 變化: {f1_int8 - f1_fp32:+.4f})")

    latency_int8 = {}
    for bs in BATCH_SIZES:
        ms = benchmark_latency(model_int8, bs, cpu_device)
        latency_int8[bs] = ms
        print(f"INT8 CPU 推論延遲 (batch={bs:3d}): {ms:.4f} ms/window")

    return {
        "latent_dim": latent_dim,
        "n_params": n_params,
        "size_fp32_mb": size_fp32,
        "size_int8_mb": size_int8_actual,
        "f1_fp32": f1_fp32,
        "f1_int8": f1_int8,
        "latency_fp32": latency_fp32,
        "latency_int8": latency_int8,
    }


def main():
    cpu_device = torch.device("cpu")
    print("=== 邊緣可行性驗證開始 (強制使用 CPU，模擬邊緣裝置環境) ===")

    test_loader = load_test_loader()

    results = []
    for dim in DIMS_TO_BENCHMARK:
        res = benchmark_one_dim(dim, test_loader, cpu_device)
        results.append(res)

    print("\n" + "=" * 100)
    print(" 邊緣可行性驗證總結")
    print("=" * 100)
    header = f"{'Dim':<6}{'Params':<10}{'FP32(MB)':<10}{'INT8(MB)':<10}{'FP32 F1':<10}{'INT8 F1':<10}{'FP32 lat(bs=1)':<16}{'INT8 lat(bs=1)':<16}"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['latent_dim']:<6}{r['n_params']:<10,}{r['size_fp32_mb']:<10.3f}{r['size_int8_mb']:<10.3f}"
            f"{r['f1_fp32']:<10.4f}{r['f1_int8']:<10.4f}"
            f"{r['latency_fp32'][1]:<16.4f}{r['latency_int8'][1]:<16.4f}"
        )
    print("=" * 100)
    print("\n單位：模型大小 MB、延遲 ms/window（batch=1，模擬邊緣單筆即時推論）")


if __name__ == "__main__":
    main()
