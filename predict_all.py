import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CSV_PATH = "dcview_camera_filtered_Outlier.csv"

ROLLING_DAYS = 14
FORECAST_MONTHS = 12
YMAX = 250_000

# -------------------------
# 工具函式
# -------------------------
def categorize_body(model: str) -> str:
    m = str(model).upper()
    if m.startswith("Z"):
        return "Z"
    if m.startswith("D"):
        return "D"
    return "OTHER"

def make_display_label(model: str, gen_int: int) -> str:
    if str(model).lower() == "z63":
        return "z63"
    return f"{model} Gen{gen_int}"

def add_features(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame("price").copy()
    df["t"] = np.arange(len(df))
    df["t2"] = df["t"] ** 2

    df["lag_1"] = df["price"].shift(1)
    df["lag_2"] = df["price"].shift(2)
    df["lag_3"] = df["price"].shift(3)

    df["ma_3"] = df["price"].rolling(3).mean()
    df["ma_6"] = df["price"].rolling(6).mean()

    return df.dropna()

FEATURES = ["t", "t2", "lag_1", "lag_2", "lag_3", "ma_3", "ma_6"]

# -------------------------
# 讀資料
# -------------------------
df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "date", "price"]).copy()
df["generation_int"] = df["generation"].round().fillna(1).astype(int)

df["body_type"] = df["camera_model"].apply(categorize_body)
df["display_label"] = [
    make_display_label(m, g)
    for m, g in zip(df["camera_model"], df["generation_int"])
]

# -------------------------
# 日資料（畫點 + 趨勢）
# -------------------------
daily = (
    df.groupby(["display_label", "body_type", "date"], as_index=False)["price"]
      .median()
      .sort_values(["display_label", "date"])
)

# -------------------------
# 準備畫布
# -------------------------
fig, ax = plt.subplots(figsize=(13, 7))
plt.subplots_adjust(right=0.78)

ax.set_title("Camera Price Trend + ML Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price (TWD)")
ax.grid(True)
ax.set_ylim(top=YMAX)

artists = {}
forecast_artists = {}

labels = sorted(daily["display_label"].unique(), key=str.lower)

# -------------------------
# 對每個機型：畫圖 + 訓練 ML + 預測
# -------------------------
for label in labels:
    g = daily[daily["display_label"] == label].sort_values("date")

    # points
    sc = ax.scatter(g["date"], g["price"], s=40, alpha=0.6)

    # trend
    g["trend"] = g["price"].rolling(
        window=ROLLING_DAYS, center=True, min_periods=1
    ).median()

    ln, = ax.plot(g["date"], g["trend"], linewidth=2.2)

    # ---------- ML 預測（月資料） ----------
    monthly = (
        g.set_index("date")["price"]
         .resample("MS").median()
         .interpolate(limit_direction="both")
    )

    if len(monthly) >= 10:
        feat = add_features(monthly)
        X = feat[FEATURES]
        y = feat["price"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0))
        ])
        model.fit(X, y)

        history = monthly.copy()
        future_vals = []

        for _ in range(FORECAST_MONTHS):
            tmp = add_features(history)
            last = tmp.iloc[-1][FEATURES].to_frame().T
            next_price = float(model.predict(last)[0])
            future_vals.append(next_price)
            history.loc[history.index.max() + pd.offsets.MonthBegin(1)] = next_price

        future_idx = pd.date_range(
            start=monthly.index.max() + pd.offsets.MonthBegin(1),
            periods=FORECAST_MONTHS,
            freq="MS"
        )

        fl, = ax.plot(
            future_idx, future_vals,
            linestyle="--", linewidth=2.4
        )
    else:
        fl = None

    artists[label] = {
        "line": ln,
        "scatter": sc,
        "body_type": g["body_type"].iloc[0]
    }
    forecast_artists[label] = fl

# -------------------------
# 勾選介面
# -------------------------
panel_labels = ["Show points", "Show forecast", "D 機身", "Z 機身"] + labels
actives = [True, True, True, True] + [True] * len(labels)

rax = fig.add_axes([0.80, 0.08, 0.19, 0.84])
check = CheckButtons(rax, panel_labels, actives)

state = {
    "show_points": True,
    "show_forecast": True,
    "show_D": True,
    "show_Z": True,
    "model_on": {lab: True for lab in labels}
}

def apply_visibility():
    for lab in labels:
        body = artists[lab]["body_type"]
        cat_ok = (body != "D" or state["show_D"]) and (body != "Z" or state["show_Z"])
        on = state["model_on"][lab] and cat_ok

        artists[lab]["line"].set_visible(on)
        artists[lab]["scatter"].set_visible(on and state["show_points"])

        if forecast_artists[lab] is not None:
            forecast_artists[lab].set_visible(on and state["show_forecast"])

    fig.canvas.draw_idle()

def on_click(label):
    if label == "Show points":
        state["show_points"] = not state["show_points"]
    elif label == "Show forecast":
        state["show_forecast"] = not state["show_forecast"]
    elif label == "D 機身":
        state["show_D"] = not state["show_D"]
    elif label == "Z 機身":
        state["show_Z"] = not state["show_Z"]
    else:
        state["model_on"][label] = not state["model_on"][label]

    apply_visibility()

check.on_clicked(on_click)
apply_visibility()

plt.show()
