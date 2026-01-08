import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# 參數設定
# =========================
CSV_PATH = "dcview_camera_filtered_Outlier.csv"

TARGET_MODEL = "Z6"
TARGET_GEN = 2

SEQ_LEN = 6          # 用過去 6 個月預測下一個月
FORECAST_MONTHS = 12
YMAX = 200_000

# =========================
# 1️⃣ 讀資料
# =========================
df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "date", "price"])

df = df[
    (df["camera_model"] == TARGET_MODEL) &
    (df["generation"] == TARGET_GEN)
].copy()

df = df.sort_values("date").set_index("date")

# =========================
# 2️⃣ 轉成月中位數（這就是你缺的 monthly）
# =========================
monthly = (
    df["price"]
    .resample("MS")
    .median()
    .interpolate(limit_direction="both")
)

print(f"Monthly samples: {len(monthly)}")

# =========================
# 3️⃣ 準備 LSTM 序列資料
# =========================
values = monthly.values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i + SEQ_LEN])
    y.append(scaled[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (samples, timesteps, features)

# =========================
# 4️⃣ 建立 LSTM 模型
# =========================
model = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)),
    Dense(1)
])

from tensorflow.keras.losses import Huber

model.compile(optimizer="adam", loss=Huber(delta=1.0))

model.summary()

# =========================
# 5️⃣ 訓練
# =========================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5)
]

model.fit(
    X, y,
    epochs=400,             # 可以開大，反正會自動早停
    batch_size=4,
    validation_split=0.2,   # 留 20% 當驗證
    shuffle=False,          # 時序資料不要洗牌
    callbacks=callbacks,
    verbose=1
)

# =========================
# 6️⃣ 遞歸預測未來 12 個月
# =========================
future_scaled = []
current_seq = scaled[-SEQ_LEN:].copy()

for _ in range(FORECAST_MONTHS):
    pred = model.predict(
        current_seq.reshape(1, SEQ_LEN, 1),
        verbose=0
    )
    future_scaled.append(pred[0, 0])
    current_seq = np.vstack([current_seq[1:], pred])

future_prices = scaler.inverse_transform(
    np.array(future_scaled).reshape(-1, 1)
).flatten()

last_real_date = monthly.index[-1]
last_real_price = float(monthly.iloc[-1])

future_idx = pd.date_range(
    start=monthly.index.max() + pd.offsets.MonthBegin(1),
    periods=FORECAST_MONTHS,
    freq="MS"
)
# ✅ 加 anchor：讓虛線第一個點就是最後真實點
plot_x = [last_real_date] + list(future_idx)
plot_y = [last_real_price] + list(future_prices)

# =========================
# 7️⃣ 視覺化
# =========================
plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly.values, label="Historical (Monthly Median)")
plt.plot(
    plot_x,
    plot_y,
    linestyle="--",
    linewidth=2.5,
    label="LSTM Forecast (anchored)"
)


plt.title(f"LSTM Price Forecast – {TARGET_MODEL} Gen{TARGET_GEN}")
plt.xlabel("Date")
plt.ylabel("Price (TWD)")
plt.grid(True)
plt.ylim(top=YMAX)
plt.legend()
plt.tight_layout()
plt.ylim(0, YMAX)
plt.show()
