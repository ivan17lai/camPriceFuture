import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dcview_camera_filtered.csv"

#SELECT_MODELS = ["Z6", "Z5", "ZF", "ZF", "Zfc", "z7", "Z8", "Z9"]
SELECT_MODELS = ["Z6"]

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "date", "price"]).copy()

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
    plt.plot(g["date"], g["price"], marker="o", linewidth=1.5, label=model_gen)

plt.xlabel("Date")
plt.ylabel("Price (TWD)")
plt.title("DCView Camera Price Trend (Separated by Generation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
