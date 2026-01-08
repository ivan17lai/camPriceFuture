import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CSV_PATH = "dcview_camera_filtered.csv"

TARGET_MODEL = "Z9"
TARGET_GEN = 1

FORECAST_MONTHS = 12
YMAX = 250_000

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "date", "price"]).copy()
df = df[(df["camera_model"] == TARGET_MODEL) & (df["generation"] == TARGET_GEN)].copy()

# 月中位數序列
df = df.sort_values("date").set_index("date")
monthly = df["price"].resample("MS").median().interpolate(limit_direction="both")
series = monthly.to_frame(name="price")

def add_features(sdf: pd.DataFrame) -> pd.DataFrame:
    out = sdf.copy()
    out["t"] = np.arange(len(out))  # 時間索引（0,1,2,...）
    out["t2"] = out["t"] ** 2       # 二次項，讓曲線更自然（可拿掉變直線）

    out["lag_1"] = out["price"].shift(1)
    out["lag_2"] = out["price"].shift(2)
    out["lag_3"] = out["price"].shift(3)
    out["ma_3"]  = out["price"].rolling(3).mean()
    out["ma_6"]  = out["price"].rolling(6).mean()
    out["ret_1"] = out["price"].pct_change(1)
    out["ret_3"] = out["price"].pct_change(3)
    return out

feat = add_features(series).dropna()

FEATURES = ["t", "t2", "lag_1", "lag_2", "lag_3", "ma_3", "ma_6", "ret_1", "ret_3"]

X = feat[FEATURES]
y = feat["price"]

# ✅ ML 模型：標準化 + Ridge（能外推趨勢，比樹模型不容易變直線）
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=3.0))
])
model.fit(X, y)

# 遞歸預測
history = series.copy()
future_idx = pd.date_range(
    start=history.index.max() + pd.offsets.MonthBegin(1),
    periods=FORECAST_MONTHS,
    freq="MS"
)

future_vals = []
for _ in future_idx:
    tmp = add_features(history).dropna()
    last_row = tmp.iloc[-1][FEATURES].to_frame().T
    next_price = float(model.predict(last_row)[0])

    future_vals.append(next_price)
    # append 下一個月
    next_date = history.index.max() + pd.offsets.MonthBegin(1)
    history.loc[next_date, "price"] = next_price

forecast = pd.Series(future_vals, index=future_idx, name="forecast")

# 畫圖
fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(series.index, series["price"], linewidth=2.2, label=f"{TARGET_MODEL} Gen{TARGET_GEN} (monthly median)")
ax.plot(forecast.index, forecast.values, linestyle="--", linewidth=2.8, label=f"ML Forecast next {FORECAST_MONTHS} months (Ridge+t+t²)")
ax.set_title("Camera Price Forecast (ML Recursive)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (TWD)")
ax.grid(True)
ax.set_ylim(top=YMAX)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
