import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ===== 中文字型（避免亂碼）=====
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 參數設定 =====
CSV_PATH = "dcview_camera_filtered_Outlier.csv"

TARGET_MODEL = "Z6"     # 你要分析的機型
TARGET_GEN = 2          # 世代，例如 Z6 Gen2

TEST_RATIO = 0.2        # 最後 20% 作為測試集
POLY_DEGREE = 3      # 多項式階數（建議 2 或 3）
FORECAST_DAYS = 365     # 預測未來一年

# ===== 讀取資料 =====
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "price", "date"]).copy()
df["generation"] = df["generation"].astype(int)

# ===== 選定單一機型 + 世代 =====
df_m = df[
    (df["camera_model"] == TARGET_MODEL) &
    (df["generation"] == TARGET_GEN)
].copy()

df_m = df_m.sort_values("date")

if len(df_m) < 30:
    raise ValueError("資料筆數過少，不適合進行預測")

# ===== 建立時間索引特徵 =====
df_m["time_index"] = np.arange(len(df_m))
X = df_m[["time_index"]].values
y = df_m["price"].values

# ===== 時間序列切分（避免資料洩漏）=====
split = int(len(df_m) * (1 - TEST_RATIO))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
date_train, date_test = df_m["date"].iloc[:split], df_m["date"].iloc[split:]

# =========================================================
# 模型一：線性回歸
# =========================================================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)
mae_lin = mean_absolute_error(y_test, y_pred_lin)

# =========================================================
# 模型二：多項式回歸
# =========================================================
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
    ("reg", LinearRegression())
])

poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

# ===== 印出誤差比較 =====
print(f"Target: {TARGET_MODEL} Gen{TARGET_GEN}")
print(f"MAE (Linear Regression): {mae_lin:.2f}")
print(f"MAE (Polynomial Regression, degree={POLY_DEGREE}): {mae_poly:.2f}")

# =========================================================
# 預測未來一年（365 天）
# =========================================================
last_index = X.max()

future_X = np.arange(
    last_index + 1,
    last_index + FORECAST_DAYS + 1
).reshape(-1, 1)

future_dates = pd.date_range(
    start=df_m["date"].iloc[-1],
    periods=FORECAST_DAYS + 1,
    freq="D"
)[1:]

future_price_lin = lin_model.predict(future_X)
future_price_poly = poly_model.predict(future_X)

# =========================================================
# 視覺化：歷史 + 未來一年
# =========================================================
plt.figure(figsize=(12, 6))

plt.plot(df_m["date"], y, label="Historical Price", alpha=0.6)
plt.plot(future_dates, future_price_lin,
         label=f"Linear Regression Forecast (MAE={mae_lin:.0f})",
         linewidth=2)

plt.plot(future_dates, future_price_poly,
         label=f"Polynomial Regression Forecast (degree={POLY_DEGREE}, MAE={mae_poly:.0f})",
         linewidth=2)

plt.title(f"{TARGET_MODEL} Gen{TARGET_GEN} 未來一年價格預測")
plt.xlabel("日期")
plt.ylabel("價格")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
