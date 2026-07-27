"""
公平對照消融測試

先前的單特徵測試用 Logistic Regression（線性、只用平均值），
跟全特徵 XGBoost（非線性、完整 128 維）不是同一個模型量級，
差距裡混雜了「模型能力」與「特徵資訊量」兩個因素。

這裡改用同一個模型（XGBoost）做三組對照，把「時序/其他特徵的邊際貢獻」單獨隔離出來：
  A. 只用 SizeKB_log 的完整 32 個時間步（不平均，保留組內變化）
  B. 只用 SizeKB_log + IsWrite 的完整時間步（Top 10 重要特徵主要落在這兩個）
  C. 全部 4 個特徵 × 32 時間步（128 維，即先前的 XGBoost baseline）

使用方式：
    python fair_comparison.py
"""

from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report


PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")

# 特徵欄位順序: ['DeltaLBA_log', 'DeltaTime_log', 'SizeKB_log', 'IsWrite']
FEATURE_NAMES = ["DeltaLBA_log", "DeltaTime_log", "SizeKB_log", "IsWrite"]
SIZEKB_IDX = 2
ISWRITE_IDX = 3


def run_xgb(X_train, y_train, X_test, y_test, name: str):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {name} ===")
    print(f"特徵維度: {X_train.shape[1]}")
    print(f"F1  : {f1:.4f}")
    print(f"AUC : {auc:.4f}")

    return {"name": name, "dim": X_train.shape[1], "f1": f1, "auc": auc}


def main():
    print("載入已處理好的資料...")
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_train = data["train_X"].numpy()
    y_train = data["train_y"].numpy()
    X_test = data["test_X"].numpy()
    y_test = data["test_y"].numpy()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    results = []

    # A. 只用 SizeKB_log 完整 32 個時間步
    Xa_train = X_train[:, :, SIZEKB_IDX]
    Xa_test = X_test[:, :, SIZEKB_IDX]
    results.append(run_xgb(Xa_train, y_train, Xa_test, y_test, "A. 僅 SizeKB_log (32 維，完整時序)"))

    # B. SizeKB_log + IsWrite 完整時間步
    Xb_train = X_train[:, :, [SIZEKB_IDX, ISWRITE_IDX]].reshape(len(X_train), -1)
    Xb_test = X_test[:, :, [SIZEKB_IDX, ISWRITE_IDX]].reshape(len(X_test), -1)
    results.append(run_xgb(Xb_train, y_train, Xb_test, y_test, "B. SizeKB_log + IsWrite (64 維)"))

    # C. 全部 4 個特徵
    Xc_train = X_train.reshape(len(X_train), -1)
    Xc_test = X_test.reshape(len(X_test), -1)
    results.append(run_xgb(Xc_train, y_train, Xc_test, y_test, "C. 全部特徵 (128 維)"))

    print("\n" + "=" * 60)
    print("公平對照總結（同一模型 XGBoost）")
    print("=" * 60)
    print(f"{'設定':<35s}{'維度':>6s}{'F1':>10s}{'AUC':>10s}")
    for r in results:
        print(f"{r['name']:<35s}{r['dim']:>6d}{r['f1']:>10.4f}{r['auc']:>10.4f}")

    gap_f1 = results[2]["f1"] - results[1]["f1"]
    gap_auc = results[2]["auc"] - results[1]["auc"]
    print(f"\nC 相對 B 的提升 (加入 DeltaLBA/DeltaTime 後): ΔF1={gap_f1:+.4f}  ΔAUC={gap_auc:+.4f}")
    if gap_f1 < 0.02 and gap_auc < 0.01:
        print("⚠️  DeltaLBA/DeltaTime 邊際貢獻很小，size+write 幾乎解釋了全部表現。")
        print("    建議調整專案敘事重點，往「壓縮效率」而非「分類準確度」發力。")
    else:
        print("✅ DeltaLBA/DeltaTime 有實質邊際貢獻，時序建模有其必要性，可放心照原計畫進行。")


if __name__ == "__main__":
    main()
