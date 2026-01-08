import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dcview_camera_filtered.csv"

SELECT_MODELS = ["Z9"]
ROLLING_DAYS = 14

TARGET_MODEL = "Z9"
TARGET_GEN = 1

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "date", "price"]).copy()

df = df[
    (df["camera_model"] == TARGET_MODEL) &
    (df["generation"] == TARGET_GEN)
].copy()

if SELECT_MODELS:
    df = df[df["camera_model"].isin(SELECT_MODELS)].copy()

df["model_gen"] = df["camera_model"] + " Gen" + df["generation"].astype(int).astype(str)

daily = (
    df.groupby(["model_gen", "date"], as_index=False)["price"]
      .median()
      .sort_values(["model_gen", "date"])
)

plt.figure(figsize=(13, 7))

for model_gen, g in daily.groupby("model_gen"):
    g = g.sort_values("date")

    # 🔹 真實資料點
    plt.scatter(
        g["date"],
        g["price"],
        s=40,
        alpha=0.6,
        label=f"{model_gen} (points)"
    )

    # 🔹 平滑趨勢線（rolling median）
    g["trend"] = g["price"].rolling(
        window=ROLLING_DAYS,
        center=True,
        min_periods=1
    ).median()

    plt.plot(
        g["date"],
        g["trend"],
        linewidth=2.2,
        label=f"{model_gen} (trend)"
    )

plt.xlabel("Date")
plt.ylabel("Price (TWD)")
plt.title("DCView Camera Price Trend (Points + Smoothed Curve)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
