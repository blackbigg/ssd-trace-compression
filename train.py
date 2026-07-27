"""
無洩漏與早停巡檢訓練腳本 (train.py)

新增修正點：
1. 加入 [2, 4] 兩組更小的 latent 維度，探索分類效能真正開始下降的臨界點。
2. 與既有 sweep_results.json 合併：已經跑過的維度不會重跑，只補上新的維度，
   避免重複花費先前 95 分鐘的訓練時間。
3. 保留切分點 Buffer 隔離、Phase 2 Early Stopping、混合重建 Loss 等既有修正。
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import DualHeadTraceModel, compute_hybrid_recon_loss


PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")
CHECKPOINT_DIR = Path(r"d:\SNIA_data\checkpoints")
RESULTS_JSON_PATH = CHECKPOINT_DIR / "sweep_results.json"

BATCH_SIZE = 256
EPOCHS_PHASE1 = 8     # 重建預訓練
EPOCHS_PHASE2 = 12    # 聯合微調 (搭配 Early Stopping)
EARLY_STOP_PATIENCE = 4  # Phase 2 耐心值
LEARNING_RATE = 1e-3
CLS_LOSS_WEIGHT = 1.0

# 新增 2、4 維，探索分類效能真正開始崩壞的臨界點
LATENT_DIMS_TO_SWEEP = [2, 4, 8, 16, 24, 32, 64]


def load_data_and_split():
    print("1. 載入持久化張量資料並做無洩漏 Validation 切分...")
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_train_raw, y_train_raw = data["train_X"], data["train_y"]
    X_test, y_test = data["test_X"], data["test_y"]

    metadata = data.get("metadata", {})
    window_size = metadata.get("window_size", 32)
    stride = metadata.get("stride", 16)

    # 算出相鄰視窗重疊的數量 (32 // 16 = 2 個視窗)
    buffer = max(window_size // stride, 1)

    val_split_idx = int(len(X_train_raw) * 0.8)

    # 切分點前後丟棄 buffer 個視窗，徹底封死邊界重疊洩漏
    X_tr = X_train_raw[: val_split_idx - buffer]
    y_tr = y_train_raw[: val_split_idx - buffer]
    X_val = X_train_raw[val_split_idx + buffer :]
    y_val = y_train_raw[val_split_idx + buffer :]

    n_pos = (y_tr == 1).sum().item()
    n_neg = (y_tr == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    print(f"   [ Buffer 隔離設定 ] Window={window_size}, Stride={stride} -> 丟棄切分點邊界前後各 {buffer} 個視窗")
    print(f"   Train_sub 視窗數: {len(X_tr):,} | Val 視窗數: {len(X_val):,} | Test 視窗數: {len(X_test):,}")
    print(f"   Train_sub 正負樣本比 pos_weight = {pos_weight.item():.4f}")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weight


def train_epoch(model, dataloader, optimizer, criterion_cls, device, is_phase_2: bool):
    model.train()
    total_loss, total_nmse, total_bce, total_cls = 0.0, 0.0, 0.0, 0.0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        x_recon_cont, logits_iswrite, logits_hot, _ = model(batch_X)

        recon_loss, nmse_val, bce_val = compute_hybrid_recon_loss(batch_X, x_recon_cont, logits_iswrite)
        cls_loss = criterion_cls(logits_hot, batch_y)

        loss = (recon_loss + CLS_LOSS_WEIGHT * cls_loss) if is_phase_2 else recon_loss

        loss.backward()
        optimizer.step()

        bs = len(batch_X)
        total_loss += loss.item() * bs
        total_nmse += nmse_val * bs
        total_bce += bce_val * bs
        total_cls += cls_loss.item() * bs

    n = len(dataloader.dataset)
    return total_loss / n, total_nmse / n, total_bce / n, total_cls / n


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, device):
    model.eval()
    total_nmse, total_bce, total_cls = 0.0, 0.0, 0.0
    all_preds, all_probs, all_targets = [], [], []

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        x_recon_cont, logits_iswrite, logits_hot, _ = model(batch_X)

        _, nmse_val, bce_val = compute_hybrid_recon_loss(batch_X, x_recon_cont, logits_iswrite)
        cls_loss = criterion_cls(logits_hot, batch_y)

        probs = torch.softmax(logits_hot, dim=1)[:, 1]
        preds = torch.argmax(logits_hot, dim=1)

        bs = len(batch_X)
        total_nmse += nmse_val * bs
        total_bce += bce_val * bs
        total_cls += cls_loss.item() * bs

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

    n = len(dataloader.dataset)
    return {
        "nmse": total_nmse / n,
        "bce": total_bce / n,
        "cls_loss": total_cls / n,
        "acc": accuracy_score(all_targets, all_preds),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "auc": roc_auc_score(all_targets, all_probs)
    }


def train_single_latent_dim(latent_dim, train_loader, val_loader, test_loader, pos_weight, device):
    print(f"\n" + "="*70)
    print(f" 開始訓練 Latent Dimension z = {latent_dim} (壓縮比: {128 / latent_dim:.1f}x)")
    print("="*70)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_model_z{latent_dim}.pt"

    model = DualHeadTraceModel(in_features=4, seq_len=32, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))

    best_val_f1 = 0.0
    patience_counter = 0
    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2

    for epoch in range(1, total_epochs + 1):
        is_phase_2 = (epoch > EPOCHS_PHASE1)
        tr_loss, tr_nmse, tr_bce, tr_cls = train_epoch(model, train_loader, optimizer, criterion_cls, device, is_phase_2)
        val_m = evaluate(model, val_loader, criterion_cls, device)

        if is_phase_2:
            if val_m['f1'] > best_val_f1 + 1e-4:
                best_val_f1 = val_m['f1']
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"   [ Early Stopping ] Phase 2 連續 {EARLY_STOP_PATIENCE} 個 epoch Val F1 未提升，提前終止當前維度訓練 (Epoch {epoch})。")
                    break

    print(f"訓練完成，最佳 Val F1 = {best_val_f1:.4f}。加載最佳權重評估 Test Set...")

    model.load_state_dict(torch.load(ckpt_path))
    test_m = evaluate(model, test_loader, criterion_cls, device)

    print(f"==> [Test 成果 z={latent_dim:2d}] F1: {test_m['f1']:.4f} | AUC: {test_m['auc']:.4f} | NMSE(cont): {test_m['nmse']:.4f} | BCE(write): {test_m['bce']:.4f}")

    return {
        "latent_dim": latent_dim,
        "compression_ratio": f"{128 / latent_dim:.1f}x",
        "test_f1": test_m['f1'],
        "test_auc": test_m['auc'],
        "test_acc": test_m['acc'],
        "test_nmse_cont": test_m['nmse'],
        "test_bce_write": test_m['bce']
    }


def load_existing_results() -> dict:
    """讀取既有的 sweep_results.json，回傳 {latent_dim: result} 的字典，方便查詢已跑過的維度。"""
    if not RESULTS_JSON_PATH.exists():
        return {}
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    return {r["latent_dim"]: r for r in existing}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 啟動無洩漏巡檢訓練管線 (Device: {device}) ===")

    existing_results = load_existing_results()
    if existing_results:
        print(f"   偵測到既有掃描結果，已完成維度: {sorted(existing_results.keys())}，將跳過重跑。")

    train_loader, val_loader, test_loader, pos_weight = load_data_and_split()

    start_total_time = time.time()
    sweep_results = dict(existing_results)  # 以字典去重，key=latent_dim

    for dim in LATENT_DIMS_TO_SWEEP:
        if dim in existing_results:
            print(f"\n[跳過] Latent Dim = {dim} 已有結果，沿用既有數據。")
            continue
        res = train_single_latent_dim(dim, train_loader, val_loader, test_loader, pos_weight, device)
        sweep_results[dim] = res

        # 每跑完一組就立刻存檔，避免中途中斷全部白費
        sorted_results = [sweep_results[d] for d in sorted(sweep_results.keys())]
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)

    total_elapsed_min = (time.time() - start_total_time) / 60.0

    final_results = [sweep_results[d] for d in sorted(sweep_results.keys())]

    print("\n" + "="*80)
    print(f" Latent Dimension 掃描帕累托前沿 (Pareto Frontier) 總結 (本次新增耗時: {total_elapsed_min:.1f} 分鐘)")
    print("="*80)
    print(f"{'Latent Dim':<12s}{'Compression':<14s}{'Test F1':<12s}{'Test AUC':<12s}{'NMSE(cont)':<14s}{'BCE(IsWrite)':<14s}")
    print("-" * 80)
    for r in final_results:
        print(f"{r['latent_dim']:<12d}{r['compression_ratio']:<14s}{r['test_f1']:<12.4f}{r['test_auc']:<12.4f}{r['test_nmse_cont']:<14.4f}{r['test_bce_write']:<14.4f}")
    print("="*80)
    print(f"\n完整掃描結果已存至: {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()