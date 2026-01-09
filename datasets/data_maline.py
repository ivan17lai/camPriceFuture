import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np



# ========= 參數 =========
INPUT_CSV = "datasets/dcview_nikon_body_no_outliers.csv"
OUTPUT_WEEKLY_CSV = "datasets/dcview_nikon_weekly_avg.csv"
TARGET_MODELS = ['Z6','Z6II', 'Z6III', 'Z7', 'Z7II', 'Z8', 'Z9', 'Z5','Z5II', 'Z50', 'Zfc']
MA_WINDOW = 14# 4 週 MA（可改 8、12）

# ========= Matplotlib 中文字體 =========
rcParams["font.family"] = "Microsoft JhengHei"
rcParams["axes.unicode_minus"] = False


def hampel_filter(series: pd.Series, window: int = 7, n_sigmas: float = 3.0):
    """
    Hampel filter：移除時間序列中的瞬時尖峰
    - window: 前後視窗大小（7 週 ≈ 1.5 個月）
    - n_sigmas: 嚴格度（3.0 很常用；2.5 更嚴）
    """
    x = series.copy()
    k = 1.4826  # MAD → std 的轉換係數

    for i in range(len(x)):
        start = max(i - window, 0)
        end = min(i + window + 1, len(x))
        window_slice = x.iloc[start:end]

        median = window_slice.median()
        mad = (window_slice - median).abs().median()

        if mad == 0 or pd.isna(mad):
            continue

        threshold = n_sigmas * k * mad

        if abs(x.iloc[i] - median) > threshold:
            # 把尖峰「壓回中位數」
            x.iloc[i] = median

    return x


# ========= 讀取資料 =========
df = pd.read_csv(INPUT_CSV)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["model", "price", "date"]).copy()
# ========= 價格硬門檻過濾（domain rule） =========
PRICE_MIN = 10_000
PRICE_MAX = 250_000

before = len(df)
df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)].copy()
after = len(df)

print(f"💡 價格過濾：移除 {before - after} 筆（< {PRICE_MIN} 或 > {PRICE_MAX}）")

# 統一 model（避免 Zf / ZF 分裂）
df["model"] = df["model"].astype(str).str.strip()
df["model_norm"] = df["model"].str.upper()
df.loc[df["model_norm"] == "ZFC", "model_norm"] = "Zfc"

# 可選：只分析部分機型
if TARGET_MODELS:
    TARGET_MODELS = [m.upper() for m in TARGET_MODELS]
    df = df[df["model_norm"].isin(TARGET_MODELS)]

# ========= 建立「週」欄位 =========
# W-MON：以週一為一週起點（分析市場比較穩）
df["week"] = df["date"].dt.to_period("W-MON").dt.to_timestamp()

# ========= 每機型 × 每週平均價格 =========
weekly_avg = (
    df.groupby(["model_norm", "week"], as_index=False)
      .agg(weekly_avg_price=("price", "mean"),
           count=("price", "size"))
      .sort_values(["model_norm", "week"])
)

# ========= 計算平滑曲線（EWMA） =========
weekly_avg["price_despiked"] = (
    weekly_avg
    .groupby("model_norm")["weekly_avg_price"]
    .transform(lambda s: hampel_filter(s, window=7, n_sigmas=3.0))
)

weekly_avg["ma_price"] = (
    weekly_avg
    .groupby("model_norm")["price_despiked"]
    .transform(lambda s: s.ewm(span=MA_WINDOW, adjust=False).mean())
)



# ========= 輸出新資料集 =========
weekly_avg.to_csv(OUTPUT_WEEKLY_CSV, index=False, encoding="utf-8-sig")

print("✅ 已建立每機型 × 每週均價資料集")
print(f"輸出檔案：{OUTPUT_WEEKLY_CSV}")
print(f"MA 週期：{MA_WINDOW} 週")
print(f"總筆數：{len(weekly_avg)}")

# ========= 視覺化（MA 均線） =========
plt.figure(figsize=(12, 6))

for model, g in weekly_avg.groupby("model_norm"):
    plt.plot(g["week"], g["ma_price"], label=model)

plt.xlabel("週")
plt.ylabel("價格（TWD）")
plt.ylim(0,250000)
plt.title(f"Nikon 各機型二手價格趨勢（{MA_WINDOW} 週移動平均）")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
