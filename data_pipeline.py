"""
MSR Cambridge Trace 資料前處理 Pipeline

修正重點（相較於前一版）：
1. 特徵計算（DeltaLBA / DeltaTimeUS / IsHot）：按 (Hostname, DiskNumber) 分組獨立計算，
   避免跨磁碟產生無意義的 Delta 跳動與熱冷標籤誤判。
2. 滑動視窗切割：同樣按磁碟分組後才切窗，避免一個視窗內混入多顆磁碟交錯的請求，
   確保模型看到的是「同一顆磁碟的真實存取序列」。
3. Train/Test 切分：改為「每顆磁碟各自按時間切 80/20」再合併，
   避免某顆磁碟資料集中在早期或晚期時段，導致 train/test 對各磁碟覆蓋不均。
4. StandardScaler：只在合併後的 train 集上 fit，test 集嚴格 transform，無資料洩漏。
5. 小磁碟保護：樣本數不足 window_size 的磁碟會被跳過並印出警告，不會讓程式崩潰。
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 加上這兩行修復中文亂碼 ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定微軟正黑體
plt.rcParams['axes.unicode_minus'] = False              # 修正負號顯示異常

# ==========================================
# 1. 高效自監督熱冷標籤生成器 (O(N) 反向掃描)
# ==========================================
def generate_hot_labels_fast(lba_array: np.ndarray, future_lookahead: int = 100) -> np.ndarray:
    """
    對單一獨立磁碟的 LBA 陣列進行自監督熱冷標籤生成，複雜度 O(N)。
    輸入必須是「同一顆磁碟、依時間排序」的 LBA 序列，否則標籤會失去意義。
    """
    n = len(lba_array)
    hot_labels = np.zeros(n, dtype=np.int64)
    block_lba = lba_array // 128  # 128 sectors * 512B = 64 KB block

    last_seen = {}
    for i in range(n - 1, -1, -1):
        blk = block_lba[i]
        if blk in last_seen and (last_seen[blk] - i) <= future_lookahead:
            hot_labels[i] = 1
        last_seen[blk] = i

    return hot_labels


# ==========================================
# 2. MSR Trace 特徵處理管線 (按磁碟隔離)
# ==========================================
def process_msr_trace(file_path, nrows: int = None, future_lookahead: int = 100) -> pd.DataFrame:
    col_names = ["Timestamp", "Hostname", "DiskNumber", "Type", "Offset", "Size", "ResponseTime"]

    print(f"1. 載入原始資料 ({file_path}, nrows={nrows})...")
    start_time = time.time()
    # Hostname/Type 用 category 讀入：字串值重複度極高，category 只存少量唯一值 + 整數編碼，
    # 對 2000 萬筆等級的資料能省下數百 MB 到 GB 等級的記憶體。
    df = pd.read_csv(
        file_path,
        compression="gzip",
        header=None,
        names=col_names,
        nrows=nrows,
        dtype={"Hostname": "category", "Type": "category"},
    )

    unique_disks = df[["Hostname", "DiskNumber"]].drop_duplicates()
    n_disks = len(unique_disks)
    print(f"   [磁碟檢查] 發現該檔案包含 {n_disks} 個獨立磁碟/Volume:")
    disk_counts = df.groupby(["Hostname", "DiskNumber"], observed=True).size()
    for (host, disk), n_rows in disk_counts.items():
        print(f"   - Host: {host}, Disk: {disk}, 請求數: {n_rows:,}")

    print("2. 基礎欄位轉換 (LBA, IsWrite, SizeKB)，並釋放不再需要的原始欄位...")
    df["LBA"] = (df["Offset"] // 512).astype(np.int64)
    df["IsWrite"] = (df["Type"] == "Write").astype(np.int8)
    df["SizeKB"] = (df["Size"] / 1024.0).astype(np.float32)
    # Offset/Size/Type 算完衍生欄位後不再需要，及早釋放降低尖峰記憶體
    df.drop(columns=["Offset", "Size", "Type"], inplace=True)
    gc.collect()

    if n_disks == 1:
        # 單磁碟特例：跳過 groupby -> concat 的多次整表複製，直接原地排序運算，
        # 大幅降低尖峰記憶體用量（這是先前 MemoryError 的主因）。
        print("3. 僅偵測到單一磁碟，採用原地排序運算路徑...")
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["DeltaLBA"] = df["LBA"].diff().fillna(0).astype(np.float32)
        df["DeltaTimeUS"] = (df["Timestamp"].diff().fillna(0) / 10.0).astype(np.float32)
        df["IsHot"] = generate_hot_labels_fast(df["LBA"].values, future_lookahead=future_lookahead)
    else:
        print("3. 按 (Hostname, DiskNumber) 分組獨立計算 Delta 特徵與 IsHot 標籤...")
        processed_groups = []
        for (host, disk), group in df.groupby(["Hostname", "DiskNumber"], sort=False, observed=True):
            group = group.sort_values("Timestamp")

            group["DeltaLBA"] = group["LBA"].diff().fillna(0).astype(np.float32)
            group["DeltaTimeUS"] = (group["Timestamp"].diff().fillna(0) / 10.0).astype(np.float32)
            group["IsHot"] = generate_hot_labels_fast(group["LBA"].values, future_lookahead=future_lookahead)

            processed_groups.append(group)

        df = pd.concat(processed_groups, axis=0, copy=False)
        del processed_groups
        gc.collect()
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    print("4. 對數縮放 (Log Transformations)...")
    df["DeltaLBA_log"] = (np.sign(df["DeltaLBA"]) * np.log1p(np.abs(df["DeltaLBA"]))).astype(np.float32)
    df["DeltaTime_log"] = np.log1p(np.maximum(0, df["DeltaTimeUS"])).astype(np.float32)
    df["SizeKB_log"] = np.log1p(df["SizeKB"]).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"=== 特徵處理完成！總耗時: {elapsed:.2f} 秒 ===")

    hot_ratio = df["IsHot"].mean()
    print(f"整體熱冷標籤分布: Hot = {hot_ratio*100:.2f}%, Cold = {(1-hot_ratio)*100:.2f}%\n")

    return df


# ==========================================
# 3. 每顆磁碟各自按時間切 80/20，避免覆蓋不均
# ==========================================
def chronological_split_per_disk(df: pd.DataFrame, train_ratio: float = 0.8):
    train_parts, test_parts = [], []

    for (host, disk), group in df.groupby(["Hostname", "DiskNumber"], sort=False, observed=True):
        group = group.sort_values("Timestamp")
        split_idx = int(len(group) * train_ratio)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    df_train = pd.concat(train_parts, axis=0).sort_values("Timestamp").reset_index(drop=True)
    df_test = pd.concat(test_parts, axis=0).sort_values("Timestamp").reset_index(drop=True)

    return df_train, df_test


# ==========================================
# 4. PyTorch Dataset (視窗切割按磁碟隔離)
# ==========================================
class TraceDataset(Dataset):
    FEATURE_COLS = ["DeltaLBA_log", "DeltaTime_log", "SizeKB_log", "IsWrite"]
    CONTINUOUS_COLS = ["DeltaLBA_log", "DeltaTime_log", "SizeKB_log"]

    def __init__(self, df: pd.DataFrame, window_size: int = 32, stride: int = 16, scaler=None):
        self.window_size = window_size
        self.stride = stride

        df = df.copy()

        # 1. 標準化：在整個 df（可能含多顆磁碟）的列層級上 fit/transform，
        #    scaler 本身不需要按磁碟分開，只有「切窗」才需要。
        if scaler is None:
            self.scaler = StandardScaler()
            df[self.CONTINUOUS_COLS] = self.scaler.fit_transform(df[self.CONTINUOUS_COLS])
        else:
            self.scaler = scaler
            df[self.CONTINUOUS_COLS] = self.scaler.transform(df[self.CONTINUOUS_COLS])

        # 2. 切窗：按磁碟分組，組內才是真實連續的存取序列
        X_list, y_list = [], []
        skipped = []

        for (host, disk), group in df.groupby(["Hostname", "DiskNumber"], sort=False, observed=True):
            group = group.sort_values("Timestamp")

            if len(group) < window_size:
                skipped.append((host, disk, len(group)))
                continue

            feats = group[self.FEATURE_COLS].values.astype(np.float32)
            labels = group["IsHot"].values.astype(np.int64)

            num_samples = (len(group) - window_size) // stride + 1
            for i in range(num_samples):
                start = i * stride
                end = start + window_size
                X_list.append(feats[start:end])

                window_hot_ratio = labels[start:end].mean()
                y_list.append(1 if window_hot_ratio >= 0.5 else 0)

        if skipped:
            print(f"   [警告] {len(skipped)} 個磁碟樣本數 < window_size({window_size})，已略過：")
            for host, disk, n in skipped:
                print(f"     - Host {host}, Disk {disk}: 僅 {n} 筆")

        if len(X_list) == 0:
            raise ValueError(
                "沒有任何磁碟產生足夠的視窗樣本，請確認 window_size 是否過大，或資料量是否過小。"
            )

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(np.array(y_list), dtype=torch.long)     # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def save_processed_data(train_dataset, test_dataset, save_path, future_lookahead: int = 100):
    """將 Train / Test Dataset 與完整的 Metadata 包裝存檔"""
    payload = {
        "train_X": train_dataset.X,
        "train_y": train_dataset.y,
        "test_X": test_dataset.X,
        "test_y": test_dataset.y,
        "scaler": train_dataset.scaler,  # 僅保存 Train 的 Scaler
        "metadata": {
            "window_size": train_dataset.window_size,
            "stride": train_dataset.stride,
            "future_lookahead": future_lookahead,
            "feature_names": TraceDataset.FEATURE_COLS,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "split_method": "chronological_per_disk",
        },
    }
    torch.save(payload, save_path)
    print(f"\n[持久化完成] 已將無洩漏的 Train/Test 資料集與 Metadata 成功儲存至:\n -> {save_path}")


# ==========================================
# 5. 主程式：磁碟隔離特徵處理 -> 磁碟隔離時序切分 -> 磁碟隔離切窗 -> 存檔
# ==========================================
if __name__ == "__main__":
    data_dir = Path(r"d:\SNIA_data")
    file_path = data_dir / "proj_2.csv.gz"
    output_pt_path = data_dir / "processed_proj_2_split.pt"

    WINDOW_SIZE = 32
    STRIDE = 16
    FUTURE_LOOKAHEAD = 100

    # 建議先用 nrows 設一個子集（例如 500 萬筆）跑通整個流程、確認記憶體夠用，
    # 再逐步調大到 None（全量 2,364 萬筆）。若仍遇到 MemoryError，
    # 表示機器實體 RAM 對全量資料不夠，直接用子集訓練即可，portfolio 專案不需要用滿全量。
    NROWS = None  # 例如先設 5_000_000 測試，穩定後再改成 None

    # 1. 特徵處理（磁碟內獨立計算 Delta / IsHot）
    df_processed = process_msr_trace(file_path, nrows=NROWS, future_lookahead=FUTURE_LOOKAHEAD)

    # 2. 每顆磁碟各自按時間切 80/20，再合併
    df_train, df_test = chronological_split_per_disk(df_processed, train_ratio=0.8)
    print(f"時序切分完成: 訓練集 raw 筆數 = {len(df_train):,}, 測試集 raw 筆數 = {len(df_test):,}")

    # 3. 建立 Train Dataset（磁碟內切窗，fit StandardScaler）
    train_dataset = TraceDataset(df_train, window_size=WINDOW_SIZE, stride=STRIDE, scaler=None)

    # 4. 建立 Test Dataset（磁碟內切窗，複用 Train 的 Scaler，防止洩漏）
    test_dataset = TraceDataset(df_test, window_size=WINDOW_SIZE, stride=STRIDE, scaler=train_dataset.scaler)

    print(f"最終產出視窗數: Train 視窗 = {len(train_dataset):,} 個, Test 視窗 = {len(test_dataset):,} 個")

    # 5. 打包持久化儲存
    save_processed_data(train_dataset, test_dataset, output_pt_path, future_lookahead=FUTURE_LOOKAHEAD)

    # 6. DataLoader 測試
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_X, batch_y in train_loader:
        print("\n=== DataLoader 正式驗證 (Train Batch) ===")
        print(f"Train 特徵 Batch 形狀 (B, T, F): {batch_X.shape}")  # [64, 32, 4]
        print(f"Train 熱冷比例 (Batch Hot Ratio): {batch_y.float().mean().item():.2f}")
        break