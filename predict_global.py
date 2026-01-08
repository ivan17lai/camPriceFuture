import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import mplcursors


# =========================
# 參數
# =========================
CSV_PATH = "dcview_camera_filtered_Outlier.csv"

TARGET_MODELS = ["Z9 Gen1", "Z8 Gen1", "Z7 Gen1", "Z7 Gen2", "Z6 Gen1", "Z6 Gen2", "Z6 Gen3", "Z5 Gen1", "Z5 Gen2", "Z50 Gen1", "ZFC Gen1"]
FORECAST_MONTHS = 12
YMAX = 150_000

LAGS = [1, 2, 3]
MA_WINDOWS = [3, 6]
VALID_LAST_N_MONTHS = 6


# =========================
# 工具：family（忽略尾巴 GenX）
# =========================
def infer_family(model_key: str) -> str:
    s = str(model_key).strip().upper()
    s = re.sub(r"\s*GEN\d+\s*$", "", s)  # ✅ 移除尾端 GenX
    m = re.match(r"^([A-Z]+)(\d)", s)
    if m:
        return m.group(1) + m.group(2)
    return s

def months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def normalize_model_name(x: str) -> str:
    # 只做簡單統一（你若有更多命名噪聲可再擴充）
    s = str(x).strip()
    return s

def remove_outliers_mad(series: pd.Series, window=6, z=4.0) -> pd.Series:
    """
    使用 rolling median + MAD 移除離群值
    只用於視覺化，不影響模型
    """
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    mad = mad.replace(0, np.nan)

    score = 0.6745 * (series - med).abs() / mad
    return series.where(score <= z)
def smooth_curve(series: pd.Series, roll=3, ema=0.35) -> pd.Series:
    """
    視覺化用平滑：
    1) rolling median 抗離群
    2) EMA 讓曲線圓滑
    """
    s = series.copy()
    s = s.rolling(roll, min_periods=1, center=True).median()
    s = s.ewm(alpha=ema, adjust=False).mean()
    return s

# =========================
# 1) 讀資料
# =========================
df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# generation 有些檔案可能沒有；用 get 保底
if "generation" in df.columns:
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["generation_int"] = df["generation"].round().astype("Int64")  # 可為 <NA>
else:
    df["generation"] = np.nan
    df["generation_int"] = pd.Series([pd.NA] * len(df), dtype="Int64")

df = df.dropna(subset=["camera_model", "date", "price"]).copy()

df["camera_model"] = df["camera_model"].apply(normalize_model_name)

# ✅ 建立 model_key：訓練/預測用的機型ID
def make_model_key(row) -> str:
    m = str(row["camera_model"]).strip()
    # 特例：z63 永遠不帶 Gen
    if m.lower() == "z63":
        return "z63"
    g = row["generation_int"]
    if pd.isna(g):
        return m
    return f"{m} Gen{int(g)}"

df["model_key"] = df.apply(make_model_key, axis=1)

df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()  # 月末 timestamp（可用）
df["family"] = df["model_key"].apply(infer_family)

# 用月資料（每月中位數）
monthly = (
    df.groupby(["model_key", "family", "month"], as_index=False)["price"]
      .median()
      .sort_values(["model_key", "month"])
)

if monthly.empty:
    raise ValueError("No data after preprocessing. Check CSV_PATH and columns.")

print("Top model_key:")
print(monthly["model_key"].value_counts().head(20))


# =========================
# 2) 推估每台第一次出現月份
# =========================
first_seen = (
    monthly.groupby("model_key")["month"]
    .min()
    .to_dict()
)

family_models = (
    monthly.groupby("family")["model_key"]
    .unique()
    .apply(list)
    .to_dict()
)


# =========================
# 3) 建立共同模型訓練表
# =========================
def build_features_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    price_map = {
        (r["model_key"], r["month"]): float(r["price"])
        for _, r in monthly_df.iterrows()
    }

    all_months_sorted = sorted(monthly_df["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months_sorted)}

    for model_key, g in monthly_df.groupby("model_key"):
        g = g.sort_values("month").copy()
        fam = g["family"].iloc[0]
        model_first = first_seen.get(model_key)

        # 自身 lag/MA
        for lag in LAGS:
            g[f"lag_{lag}"] = g["price"].shift(lag)
        for w in MA_WINDOWS:
            g[f"ma_{w}"] = g["price"].rolling(w).mean()

        # 波動
        g["ret_1m"] = g["price"].pct_change(1)
        g["vol_3m"] = g["ret_1m"].rolling(3).std()

        # 時間特徵
        g["month_idx"] = g["month"].map(month_to_idx).astype(int)
        g["months_since_first_seen"] = g["month"].apply(lambda m: float(months_diff(m, model_first)))

        # 新機衝擊（同 family）
        fam_list = family_models.get(fam, [])
        newer_models = [m for m in fam_list if first_seen.get(m) and first_seen[m] > model_first]

        newer_count_list = []
        months_since_newest_launch_list = []
        newest_price_list = []
        gap_to_newest_list = []

        for _, row in g.iterrows():
            cur_month = row["month"]

            active_newers = [m for m in newer_models if first_seen[m] <= cur_month]
            newer_count_list.append(len(active_newers))

            newest_model = None
            newest_launch = None
            if active_newers:
                newest_model = max(active_newers, key=lambda m: first_seen[m])
                newest_launch = first_seen[newest_model]

            if newest_launch is None:
                months_since_newest_launch_list.append(0.0)
                newest_price_list.append(np.nan)
                gap_to_newest_list.append(np.nan)
            else:
                months_since_newest_launch_list.append(float(months_diff(cur_month, newest_launch)))

                newest_price = price_map.get((newest_model, cur_month), np.nan)
                newest_price_list.append(newest_price)

                if np.isnan(newest_price):
                    gap_to_newest_list.append(np.nan)
                else:
                    gap_to_newest_list.append(float(row["price"] - newest_price))

        g["newer_in_family_count"] = newer_count_list
        g["months_since_newest_launch"] = months_since_newest_launch_list
        g["newest_model_price"] = newest_price_list
        g["price_gap_to_newest"] = gap_to_newest_list

        # 目標：下一個月 price
        g["target_next_price"] = g["price"].shift(-1)

        all_rows.append(g)

    out = pd.concat(all_rows, ignore_index=True)

    need_cols = [f"lag_{l}" for l in LAGS] + [f"ma_{w}" for w in MA_WINDOWS] + ["target_next_price"]
    out = out.dropna(subset=need_cols).copy()

    return out


train_df = build_features_table(monthly)

train_df["newest_model_price"] = train_df["newest_model_price"].fillna(0.0)
train_df["price_gap_to_newest"] = train_df["price_gap_to_newest"].fillna(0.0)
train_df["vol_3m"] = train_df["vol_3m"].fillna(0.0)


# =========================
# 4) 時序切分
# =========================
max_month = train_df["month"].max()
cutoff = (max_month - pd.offsets.MonthBegin(VALID_LAST_N_MONTHS))

train_part = train_df[train_df["month"] < cutoff].copy()
valid_part = train_df[train_df["month"] >= cutoff].copy()

if len(train_part) < 50:
    print("⚠️ 訓練資料偏少，建議擴增機型或增加月份資料。")


# =========================
# 5) 建共同模型
# =========================
target = "target_next_price"

num_features = [
    "month_idx",
    "months_since_first_seen",
    "ret_1m",
    "vol_3m",
    "newer_in_family_count",
    "months_since_newest_launch",
    "newest_model_price",
    "price_gap_to_newest",
] + [f"lag_{l}" for l in LAGS] + [f"ma_{w}" for w in MA_WINDOWS]

cat_features = ["model_key", "family"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features
        ),
        ("num", "passthrough", num_features),
    ]
)


reg = HistGradientBoostingRegressor(
    loss="squared_error",
    max_depth=6,
    learning_rate=0.06,
    max_iter=500,
    random_state=42
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("reg", reg)
])

X_train = train_part[cat_features + num_features]
y_train = train_part[target].astype(float)

X_valid = valid_part[cat_features + num_features]
y_valid = valid_part[target].astype(float)

model.fit(X_train, y_train)

if len(valid_part) > 0:
    pred = model.predict(X_valid)
    mae = np.mean(np.abs(pred - y_valid.values))
    print(f"✅ Global Model Validation MAE ≈ {mae:,.0f} TWD")
else:
    print("⚠️ 驗證集為空，略過 MAE。")


# =========================
# 6) 用共同模型預測多個 model_key（每台遞歸 12 個月）
# =========================
def get_monthly_series(model_key: str) -> pd.Series:
    g = monthly[monthly["model_key"] == model_key].sort_values("month")
    if g.empty:
        return pd.Series(dtype=float)
    s = g.set_index("month")["price"].copy()
    # 補齊月缺值（線性插補）
    s = s.resample("MS").median().interpolate(limit_direction="both")
    return s

def build_one_row_features(model_key: str, cur_month: pd.Timestamp, series: pd.Series) -> dict:
    fam = infer_family(model_key)
    model_first = first_seen.get(model_key, series.index.min())

    feat = {
        "model_key": model_key,
        "family": fam,
        "month_idx": int(train_df["month_idx"].max() + 1),
        "months_since_first_seen": float(months_diff(cur_month, model_first)),
        "ret_1m": float(series.pct_change(1).iloc[-1]) if len(series) >= 2 else 0.0,
        "vol_3m": float(series.pct_change(1).rolling(3).std().iloc[-1]) if len(series) >= 4 else 0.0,
    }

    for lag in LAGS:
        feat[f"lag_{lag}"] = float(series.iloc[-lag]) if len(series) >= lag else float(series.iloc[-1])

    for w in MA_WINDOWS:
        feat[f"ma_{w}"] = float(series.tail(w).mean()) if len(series) >= w else float(series.mean())

    fam_list = family_models.get(fam, [])
    model_first_seen = first_seen.get(model_key, model_first)
    newer_models = [m for m in fam_list if first_seen.get(m) and first_seen[m] > model_first_seen]
    active_newers = [m for m in newer_models if first_seen[m] <= cur_month]

    feat["newer_in_family_count"] = len(active_newers)

    newest_model = None
    newest_launch = None
    if active_newers:
        newest_model = max(active_newers, key=lambda m: first_seen[m])
        newest_launch = first_seen[newest_model]

    if newest_launch is None:
        feat["months_since_newest_launch"] = 0.0
        feat["newest_model_price"] = 0.0
        feat["price_gap_to_newest"] = 0.0
    else:
        feat["months_since_newest_launch"] = float(months_diff(cur_month, newest_launch))
        newest_price = monthly[
            (monthly["model_key"] == newest_model) &
            (monthly["month"] == cur_month)
        ]["price"]
        newest_price = float(newest_price.iloc[0]) if len(newest_price) else 0.0
        feat["newest_model_price"] = newest_price
        feat["price_gap_to_newest"] = float(series.iloc[-1] - newest_price) if newest_price != 0.0 else 0.0

    return feat


# month_idx 延伸（所有機型共用）
all_months_sorted = sorted(monthly["month"].unique())
month_to_idx = {m: i for i, m in enumerate(all_months_sorted)}
base_idx = max(month_to_idx.values()) if month_to_idx else 0

# =========================
# 7) 畫圖（多機型）
# =========================
plt.figure(figsize=(12, 6))

# 若你沒有設定 TARGET_MODELS，就用出現最多的前幾名（保底）
if "TARGET_MODELS" not in globals():
    TARGET_MODELS = monthly["model_key"].value_counts().head(5).index.tolist()

missing = []
too_short = []

for mk in TARGET_MODELS:
    s = get_monthly_series(mk)

    if len(s) == 0:
        missing.append(mk)
        continue

    if len(s) < (max(LAGS + MA_WINDOWS) + 2):
        too_short.append((mk, len(s)))
        continue

    last_real_date = s.index.max()
    last_real_price = float(s.iloc[-1])

    future_months = pd.date_range(
        start=last_real_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS"
    )
    future_idx_map = {m: base_idx + i + 1 for i, m in enumerate(future_months)}

    pred_prices = []
    series_ext = s.copy()

    for m in future_months:
        feat = build_one_row_features(mk, m, series_ext)
        feat["month_idx"] = future_idx_map[m]

        X_one = pd.DataFrame([feat])[cat_features + num_features]
        yhat = float(model.predict(X_one)[0])

        pred_prices.append(yhat)
        series_ext.loc[m] = yhat

    # ✅ 視覺化用平滑（不影響模型）
    # =========================
    # 🔗 關鍵修正：歷史 + 預測 → 一起平滑
    # =========================

    # 1️⃣ 先把歷史 + 預測接起來（注意：不要重複 last_real）
    full_series = pd.concat([
        s,
        pd.Series(pred_prices, index=future_months)
    ])

    # 2️⃣ 對「完整序列」一次平滑（EMA 記憶不斷）
    full_smooth = smooth_curve(full_series, roll=3, ema=0.35)

    # 3️⃣ 再拆回歷史 / 預測（只用來畫線）
    hist_smooth = full_smooth.loc[s.index]
    forecast_smooth = full_smooth.loc[future_months]

    # =========================
    # 4️⃣ 畫圖
    # =========================

    # 實線：歷史
    hist_line, = plt.plot(
        hist_smooth.index,
        hist_smooth.values,
        linewidth=2.2,
        label=f"{mk} (history)"
    )
    c = hist_line.get_color()

    forecast_line, = plt.plot(
        forecast_smooth.index,
        forecast_smooth.values,
        linestyle="--",
        linewidth=2.6,
        color=c,
        label=f"{mk} (forecast)"
    )



if missing:
    print("⚠️ 找不到以下機型（model_key 不存在於資料）：", missing)
if too_short:
    print("⚠️ 月資料太少，略過：", too_short)
    print("   建議：選資料較多的 model_key，或把 CSV 資料範圍拉長。")


plt.title("Global Model Forecast – Multi Models (Gen-aware)")
plt.xlabel("Date")
plt.ylabel("Price (TWD)")
plt.grid(True)
plt.ylim(top=YMAX)
plt.legend()
plt.tight_layout()

# =========================
# 8) 滑鼠 hover 顯示（時間 / 價格 / 機型 / 類型）
# =========================
cursor = mplcursors.cursor(hover=True)

@cursor.connect("add")
def _(sel):
    artist = sel.artist
    x, y = sel.target

    # x 會是 matplotlib date float，要轉回 Timestamp
    dt = pd.to_datetime(x, unit="D", origin="unix")  # 有時候不準


plt.show()
