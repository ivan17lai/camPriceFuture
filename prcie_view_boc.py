import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

CSV_PATH = "dcview_camera_filtered.csv"

# 你可以在這裡先做資料篩選（可留空）
FILTER_MODELS = []          # 例如 ["Z9", "Z6"]；空陣列 = 不限
FILTER_GENS = []            # 例如 [1, 2]；空陣列 = 不限

ROLLING_DAYS = 14

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "date", "price"]).copy()

if FILTER_MODELS:
    df = df[df["camera_model"].isin(FILTER_MODELS)].copy()

if FILTER_GENS:
    df = df[df["generation"].isin(FILTER_GENS)].copy()

df["generation_int"] = df["generation"].round().astype(int)
df["model_gen"] = df["camera_model"].astype(str) + " Gen" + df["generation_int"].astype(str)

daily = (
    df.groupby(["model_gen", "date"], as_index=False)["price"]
      .median()
      .sort_values(["model_gen", "date"])
)

# ---- 畫圖 + 互動勾選 ----
fig, ax = plt.subplots(figsize=(13, 7))
# 右側留空給勾選區
plt.subplots_adjust(right=0.78)

lines = {}      # model_gen -> Line2D
scatters = {}   # model_gen -> PathCollection

# 先畫出每個 model_gen 的 points 與 trend
for model_gen, g in daily.groupby("model_gen"):
    g = g.sort_values("date").copy()

    # points
    sc = ax.scatter(
        g["date"], g["price"],
        s=40, alpha=0.6,
        label=f"{model_gen} (points)"
    )

    # rolling median trend
    g["trend"] = g["price"].rolling(
        window=ROLLING_DAYS,
        center=True,
        min_periods=1
    ).median()

    ln, = ax.plot(
        g["date"], g["trend"],
        linewidth=2.2,
        label=f"{model_gen} (trend)"
    )

    scatters[model_gen] = sc
    lines[model_gen] = ln

ax.set_xlabel("Date")
ax.set_ylabel("Price (TWD)")
ax.set_title("DCView Camera Price Trend (Interactive)")
ax.grid(True)

# 初始狀態：全部顯示、資料點顯示
show_points = True
model_list = sorted(lines.keys())

# CheckButtons：第一個是「顯示資料點」，後面是每個 model_gen
labels = ["Show points"] + model_list
actives = [True] + [True] * len(model_list)

rax = fig.add_axes([0.80, 0.15, 0.18, 0.70])  # [left, bottom, width, height]
check = CheckButtons(rax, labels, actives)

def refresh_visibility():
    global show_points
    for mg in model_list:
        # model_gen 是否開啟
        mg_on = lines[mg].get_visible()

        # trend 永遠跟 model_gen 開關
        lines[mg].set_visible(mg_on)

        # points 受兩個開關影響：model_gen + show_points
        scatters[mg].set_visible(mg_on and show_points)

    fig.canvas.draw_idle()

def on_check(label):
    global show_points

    if label == "Show points":
        show_points = not show_points
        refresh_visibility()
        return

    # 切換某個 model_gen 的可見性
    current = lines[label].get_visible()
    lines[label].set_visible(not current)
    refresh_visibility()

check.on_clicked(on_check)

# 如果你不想要 legend（因為右側已可勾選），可以註解掉下一行
ax.legend(loc="upper left")
ax.set_ylim(0, 200_000)

plt.tight_layout()
plt.show()
