"""
手工特徵 vs 2D 神經網路 Latent 極限對照實驗 (baseline_manual_2d.py)

目的：
驗證僅使用視窗內 SizeKB_log 的 2 個統計量 (Mean, Std)，
透過 Simple Logistic Regression / XGBoost，能否達到與 z=2 神經網路相同的 F1 分數。
"""

from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import xgboost as xgb

PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")


def run_manual_feature_experiment():
    print("1. 載入資料集...")
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_train, y_train = data["train_X"], data["train_y"]
    X_test, y_test = data["test_X"], data["test_y"]

    # 原始 X 結構: (N, 32, 4)，其中 idx=2 為 SizeKB_log (假設順序為 [DeltaLBA, DeltaTime, SizeKB, IsWrite])
    # 抽取 SizeKB_log 通道
    size_kb_train = X_train[:, :, 2].numpy()
    size_kb_test = X_test[:, :, 2].numpy()

    print("2. 構建 2 維手工統計特徵 [Mean, Std]...")
    # 手工 2 維特徵：Mean 與 Std
    manual_train = np.column_stack(
        [size_kb_train.mean(axis=1), size_kb_train.std(axis=1)]
    )
    manual_test = np.column_stack(
        [size_kb_test.mean(axis=1), size_kb_test.std(axis=1)]
    )

    print("\n=== 實驗 A: 2D 手工特徵 + Logistic Regression (極簡線性模型) ===")
    clf_lr = LogisticRegression(random_state=42)
    clf_lr.fit(manual_train, y_train.numpy())
    preds_lr = clf_lr.predict(manual_test)
    probs_lr = clf_lr.predict_proba(manual_test)[:, 1]

    f1_lr = f1_score(y_test.numpy(), preds_lr, pos_label=1)
    auc_lr = roc_auc_score(y_test.numpy(), probs_lr)
    print(f"Test F1 (Hot): {f1_lr:.4f}")
    print(f"Test ROC-AUC : {auc_lr:.4f}")

    print("\n=== 實驗 B: 2D 手工特徵 + XGBoost ===")
    clf_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_xgb.fit(manual_train, y_train.numpy())
    preds_xgb = clf_xgb.predict(manual_test)
    probs_xgb = clf_xgb.predict_proba(manual_test)[:, 1]

    f1_xgb = f1_score(y_test.numpy(), preds_xgb, pos_label=1)
    auc_xgb = roc_auc_score(y_test.numpy(), probs_xgb)
    print(f"Test F1 (Hot): {f1_xgb:.4f}")
    print(f"Test ROC-AUC : {auc_xgb:.4f}")

    print("\n" + "=" * 50)
    print("【基準對照參考】")
    print("神經網路 2D Latent (z=2 Dual-Head Model) Test F1: 0.9171 | AUC: 0.9935")
    print("=" * 50)


if __name__ == "__main__":
    run_manual_feature_experiment()