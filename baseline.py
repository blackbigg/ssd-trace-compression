"""
熱冷分類 Baseline

在訓練 Conv+Attention 神經網路之前，先建立兩個對照組：
1. 多數類別 baseline（DummyClassifier）：最低門檻，證明「什麼都不做」的表現
2. XGBoost baseline：用你已經處理好的 4 個特徵（展平成 32*4=128 維向量）訓練傳統樹模型，
   這是你的神經網路必須要打贏的對手——如果神經網路贏不了 XGBoost，
   代表 Conv+Attention 架構沒有從時序結構中學到額外的東西，值得重新檢視設計。

評估指標刻意不只看 Accuracy：因為 train/test 熱冷比例本身就不平衡（39% vs 24%，
且資料存在明顯的 workload regime 切換，見 hot_ratio_trend.png 的分析），
Accuracy 容易被多數類別掩蓋，所以以 F1 / ROC-AUC / Precision / Recall 為主要判斷依據。

使用方式：
    python baseline.py
需要先安裝 xgboost：pip install xgboost
"""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("請先執行 `pip install xgboost` 再執行這個腳本。")


PROCESSED_PT_PATH = Path(r"d:\SNIA_data\processed_proj_1_split.pt")

# 若資料量太大導致 XGBoost 訓練太慢，可以設一個子抽樣上限先快速迭代，
# 例如 300_000，等確認 pipeline 沒問題後再設 None 用全量跑出最終數字。
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None

FEATURE_NAMES = ["DeltaLBA_log", "DeltaTime_log", "SizeKB_log", "IsWrite"]


def load_data():
    print("載入已處理好的資料...")
    data = torch.load(PROCESSED_PT_PATH, weights_only=False)
    X_train, y_train = data["train_X"].numpy(), data["train_y"].numpy()
    X_test, y_test = data["test_X"].numpy(), data["test_y"].numpy()
    window_size = data["metadata"]["window_size"]

    if MAX_TRAIN_SAMPLES is not None and len(X_train) > MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if MAX_TEST_SAMPLES is not None and len(X_test) > MAX_TEST_SAMPLES:
        idx = np.random.choice(len(X_test), MAX_TEST_SAMPLES, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train Hot Ratio: {y_train.mean():.4f} | Test Hot Ratio: {y_test.mean():.4f}")

    # 展平 (N, T, F) -> (N, T*F)，讓傳統 ML 模型可以直接吃這個特徵向量
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, y_train, X_test_flat, y_test, window_size


def evaluate(y_true, y_pred, y_prob, model_name: str):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # 若 y_prob 全為常數（例如 dummy baseline）會發生

    print(f"\n=== {model_name} 評估結果 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (Hot): {precision:.4f}")
    print(f"Recall    (Hot): {recall:.4f}")
    print(f"F1        (Hot): {f1:.4f}")
    print(f"ROC-AUC        : {auc:.4f}")
    print("\nConfusion Matrix (rows=實際, cols=預測, [Cold, Hot]):")
    print(confusion_matrix(y_true, y_pred))
    print("\n完整分類報告:")
    print(classification_report(y_true, y_pred, target_names=["Cold", "Hot"], zero_division=0))

    return {"model": model_name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def run_majority_baseline(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("Baseline 1: 多數類別 (Dummy Classifier)")
    print("=" * 60)

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return evaluate(y_test, y_pred, y_prob, "多數類別 Baseline")


def run_xgboost_baseline(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("Baseline 2: XGBoost")
    print("=" * 60)

    # 處理類別不平衡：scale_pos_weight = 負樣本數 / 正樣本數
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.4f} (Train 正負樣本比)")

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

    start = time.time()
    clf.fit(X_train, y_train)
    print(f"訓練耗時: {time.time() - start:.1f} 秒")

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    result = evaluate(y_test, y_pred, y_prob, "XGBoost Baseline")

    # 特徵重要性：因為輸入是展平的 (T*F)，第 i 個特徵對應第 i//F 個時間步的第 i%F 個特徵
    print("\n=== Top 10 重要特徵 (時間步, 特徵名稱) ===")
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    n_features = len(FEATURE_NAMES)
    for idx in top_idx:
        t_step = idx // n_features
        f_name = FEATURE_NAMES[idx % n_features]
        print(f"  時間步 {t_step:2d} | {f_name:16s} | 重要性 = {importances[idx]:.4f}")

    return result


def main():
    X_train, y_train, X_test, y_test, window_size = load_data()

    results = []
    results.append(run_majority_baseline(X_train, y_train, X_test, y_test))
    results.append(run_xgboost_baseline(X_train, y_train, X_test, y_test))

    print("\n" + "=" * 60)
    print("Baseline 總結比較")
    print("=" * 60)
    print(f"{'Model':<25s}{'Accuracy':>10s}{'F1':>10s}{'AUC':>10s}")
    for r in results:
        print(f"{r['model']:<25s}{r['accuracy']:>10.4f}{r['f1']:>10.4f}{r['auc']:>10.4f}")

    print(
        "\n這是你的神經網路要打贏的門檻："
        "F1 與 AUC 至少要明顯超過 XGBoost，"
        "否則代表 Conv+Attention 架構沒有從時序結構中學到額外的訊號。"
    )


if __name__ == "__main__":
    main()