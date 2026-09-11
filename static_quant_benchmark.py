import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.ao.quantization as tq
from sklearn.metrics import f1_score

from model import DualHeadTraceModel

# 1. 環境與路徑設定
PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")
CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")
LATENT_DIM = 2  # 選擇 2 維進行極致評測

# =========================================================
# 【修復重點】自動偵測 Windows 支援的量化引擎 (FBGEMM / QNNPACK / ONEDNN)
# =========================================================
supported_engines = torch.backends.quantized.supported_engines
print(f"目前 PyTorch 系統支援的量化引擎列表: {supported_engines}")

chosen_engine = None
# 優先順序: fbgemm -> qnnpack -> onednn -> x86
for eng in ['fbgemm', 'qnnpack', 'onednn', 'x86']:
    if eng in supported_engines:
        chosen_engine = eng
        break

if chosen_engine:
    torch.backends.quantized.engine = chosen_engine
    print(f"成功設定啟用量化引擎: '{chosen_engine}'")
else:
    print("警告：未偵測到指定的預編譯量化引擎，將嘗試預設流程執行。")


def load_data():
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    # 取前 10,000 筆 Train 集資料做校準 (Calibration)
    calib_loader = DataLoader(TensorDataset(data["train_X"][:10000], data["train_y"][:10000]), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(data["test_X"], data["test_y"]), batch_size=256, shuffle=False)
    return calib_loader, test_loader

@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        logits_hot = outputs[2]
        preds = torch.argmax(logits_hot, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.numpy())
    return f1_score(all_targets, all_preds, zero_division=0)

@torch.no_grad()
def benchmark_latency(model, batch_size=1, n_runs=200):
    model.eval()
    dummy = torch.randn(batch_size, 32, 4)
    # Warm-up 預熱，消除啟動開銷
    for _ in range(20):
        _ = model(dummy)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    elapsed = time.perf_counter() - start
    return (elapsed / (n_runs * batch_size)) * 1000  # ms/window

def main():
    print("\n=== 靜態量化 (Static Quantization) 實務驗證開始 ===")
    calib_loader, test_loader = load_data()

    # 1. 載入原始 FP32 模型
    ckpt_path = CHECKPOINT_DIR / f"best_model_z{LATENT_DIM}.pt"
    model_fp32 = DualHeadTraceModel(in_features=4, seq_len=32, latent_dim=LATENT_DIM)
    model_fp32.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_fp32.eval()

    # 2. 根據當前引擎取得對應的 QConfig 配置
    if chosen_engine == 'qnnpack':
        qconfig_mapping = tq.get_default_qconfig_mapping('qnnpack')
    else:
        qconfig_mapping = tq.get_default_qconfig_mapping('fbgemm')

    example_inputs = (torch.randn(1, 32, 4),)

    # 3. Prepare：自動捕捉計算圖並插入觀察器 (Observer)
    print("1. 正在進行計算圖融合與 Observer 插入 (Prepare FX)...")
    model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)

    # 4. Calibrate：灌入 10,000 筆真實 Trace 資料，記錄每層 Activation 的範圍
    print("2. 正在以 10,000 筆資料進行範圍校準 (Calibration)...")
    with torch.no_grad():
        for batch_X, _ in calib_loader:
            model_prepared(batch_X)

    # 5. Convert：轉為純整數 INT8 靜態量化模型
    print("3. 轉換為純 INT8 靜態量化模型 (Convert FX)...")
    model_static_int8 = convert_fx(model_prepared)

    # 6. 評測與效能對比
    print("\n" + "="*75)
    print(f" 靜態量化驗證結果對比 ($z = {LATENT_DIM}$, Engine: {chosen_engine})")
    print("="*75)

    # FP32 原始模型數據
    f1_fp32 = evaluate_model(model_fp32, test_loader)
    lat_fp32 = benchmark_latency(model_fp32, batch_size=1)

    # INT8 靜態量化模型數據
    f1_static = evaluate_model(model_static_int8, test_loader)
    lat_static = benchmark_latency(model_static_int8, batch_size=1)

    # 量化模型寫檔測量精準體積
    tmp_path = Path("_tmp_static_quant.pt")
    torch.save(model_static_int8.state_dict(), tmp_path)
    size_static_mb = tmp_path.stat().st_size / (1024 ** 2)
    tmp_path.unlink()

    print(f"{'指標':<20}{'FP32 原始模型':<20}{'INT8 靜態量化 (Static)':<25}")
    print("-" * 75)
    print(f"{'模型體積 (MB)':<18}{0.595:<20.3f}{size_static_mb:<25.3f}")
    print(f"{'Test F1-Score':<18}{f1_fp32:<20.4f}{f1_static:<25.4f}")
    print(f"{'CPU 延遲 (bs=1)':<17}{lat_fp32:<20.4f} ms{lat_static:<25.4f} ms")
    print("="*75)

if __name__ == "__main__":
    main()