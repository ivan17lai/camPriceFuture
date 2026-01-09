import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========= 參數 =========
K = 1.5
WINDOW_DAYS = 15          # 前後 15 天
MIN_POINTS = 5           # 視窗內至少幾筆才判離群
INPUT_CSV = "datasets/dcview_nikon_body_only.csv"
OUTPUT_CSV = "datasets/dcview_nikon_body_no_outliers.csv"
TARGET_MODEL = "Z6"

# ========= Matplotlib 中文字體（Windows）=========
rcParams["font.family"] = "Microsoft JhengHei"
rcParams["axes.unicode_minus"] = False

# ========= 讀取資料 =========
df = pd.read_csv(INPUT_CSV)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["model", "price", "date"]).copy()

# 統一 model（避免 Zf / ZF 分裂）
df["model"] = df["model"].astype(str).str.strip()
df["model_norm"] = df["model"].str.upper()
df.loc[df["model_norm"] == "ZFC", "model_norm"] = "Zfc"

# ========= 滑動視窗離群判定（同機型，±15天） =========
def mark_outliers_rolling_iqr(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    對單一機型資料，對每筆用「日期 ± WINDOW_DAYS」內的價格分布算 IQR
    判斷該筆是否離群。
    """
    m = model_df.sort_values("date").copy()
    dates = m["date"].to_numpy(dtype="datetime64[ns]")
    prices = m["price"].to_numpy(dtype=float)

    # 方便用 searchsorted 找視窗範圍
    # left = first index >= date - WINDOW_DAYS
    # right = first index >  date + WINDOW_DAYS
    is_outlier = np.zeros(len(m), dtype=bool)
    lower_arr = np.full(len(m), np.nan)
    upper_arr = np.full(len(m), np.nan)
    n_win_arr = np.zeros(len(m), dtype=int)

    # 預先把 dates 轉成 int64 ns，searchsorted 對 numpy datetime64 也可用
    for i in range(len(m)):
        left_time = dates[i] - np.timedelta64(WINDOW_DAYS, "D")
        right_time = dates[i] + np.timedelta64(WINDOW_DAYS, "D")

        left = np.searchsorted(dates, left_time, side="left")
        right = np.searchsorted(dates, right_time, side="right")

        window_prices = prices[left:right]
        n_win = len(window_prices)
        n_win_arr[i] = n_win

        # 視窗內資料太少 → 不判離群
        if n_win < MIN_POINTS:
            continue

        q1 = np.quantile(window_prices, 0.25)
        q3 = np.quantile(window_prices, 0.75)
        iqr = q3 - q1

        # IQR 太小（或 0） → 不判離群
        if iqr == 0 or np.isnan(iqr):
            continue

        lower = q1 - K * iqr
        upper = q3 + K * iqr

        lower_arr[i] = lower
        upper_arr[i] = upper
        is_outlier[i] = not (lower <= prices[i] <= upper)

    m["win_n"] = n_win_arr
    m["iqr_lower"] = lower_arr
    m["iqr_upper"] = upper_arr
    m["is_outlier"] = is_outlier
    return m

# 對每個機型做 rolling outlier 標記
df_marked = (
    df.groupby("model_norm", group_keys=False)
      .apply(mark_outliers_rolling_iqr)
)

# clean：移除離群
df_clean = df_marked[~df_marked["is_outlier"]].copy()

# 輸出 clean CSV（丟掉輔助欄位）
drop_cols = ["model_norm", "win_n", "iqr_lower", "iqr_upper", "is_outlier"]
df_clean_out = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

df_clean_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("✅ 已依『同機型 + 滑動視窗（前後 15 天）』剔除離群值")
print(f"WINDOW_DAYS=±{WINDOW_DAYS} | K={K} | MIN_POINTS={MIN_POINTS}")
print(f"原始筆數：{len(df)}")
print(f"清洗後筆數：{len(df_clean)}")
print(f"輸出檔案：{OUTPUT_CSV}")

# ========= Z6：同規則展示（X=時間, Y=價格） =========
df_z6 = df_marked[df_marked["model_norm"] == TARGET_MODEL].copy()
if df_z6.empty:
    print(f"⚠️ 找不到 {TARGET_MODEL} 的資料")
    raise SystemExit

normal = df_z6[~df_z6["is_outlier"]]
outliers = df_z6[df_z6["is_outlier"]]

print(f"\n📌 {TARGET_MODEL}（滑動視窗）總筆數：{len(df_z6)}")
print(f"正常資料：{len(normal)}")
print(f"離群值：{len(outliers)}")

if not outliers.empty:
    print("\n❌ 離群值清單（依日期/價格排序）：")
    show_cols = [c for c in ["date", "price", "win_n", "iqr_lower", "iqr_upper", "title", "region", "link"] if c in outliers.columns]
    print(outliers[show_cols].sort_values(["date", "price"]).to_string(index=False))

# 視覺化：時間 × 價格
plt.figure(figsize=(12, 6))
plt.scatter(normal["date"], normal["price"], alpha=0.7, label="正常價格")
plt.scatter(outliers["date"], outliers["price"], marker="x", s=90, label="離群值")

plt.xlabel("日期")
plt.ylabel("價格（TWD）")
plt.ylim(0, 100000)
plt.title(f"Nikon {TARGET_MODEL} 二手價格（滑動視窗 ±{WINDOW_DAYS} 天 IQR）離群值分析")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
