import pandas as pd
import matplotlib.pyplot as plt

# ===== 中文字型（避免亂碼）=====
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 參數 =====
CSV_PATH = "dcview_camera_filtered_Outlier.csv"
MA_WINDOW = 14  # 移動平均天數（可改 7 / 14 / 30）

# ===== 讀取資料 =====
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

df = df.dropna(subset=["camera_model", "generation", "price", "date"]).copy()

# 只保留 Z 系列（自動排除 D 系列）
# df = df[df["camera_model"].astype(str).str.startswith("Z")].copy()

# 產生 model_gen（例如 Z6 Gen2）
df["generation"] = df["generation"].astype(int)
df["model_gen"] = df["camera_model"] + " Gen" + df["generation"].astype(str)


def plot_series(title, df_sub, ma_window=30):
    """畫出原始資料點（scatter）並疊上 MA 均線"""
    fig, ax = plt.subplots(figsize=(12, 6))

    from itertools import cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(colors)
    for key, g in df_sub.groupby("model_gen"):
        g = g.sort_values("date")
        if g.empty:
            continue

        # 取得顏色後共用於 scatter 與 MA 線，保持視覺一致
        color = next(color_cycle)

        # 畫原始資料點（放在底層，alpha 較低）
        ax.scatter(g["date"], g["price"], s=10, alpha=0.4, color=color)

        # 以 date 做 index，計算移動平均並去除前面的 NaN
        ma = g.set_index("date")["price"].rolling(ma_window).mean().dropna()
        if ma.empty:
            continue

        ax.plot(ma.index, ma.values, linewidth=2, label=f"{key} MA{ma_window}", color=color)

    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("價格")
    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ===== 圖一：所有 Z 系列 =====
plot_series(
    title=f"圖一：所有 Z 機身價格 MA{MA_WINDOW}（離群值過濾後）",
    df_sub=df,
    ma_window=MA_WINDOW
)

# ===== 圖二：只顯示 Z50 / FC / Z6 / Z7 / Z8 =====
df_selected = df[df["camera_model"].isin(["Z50", "ZFC", "Z9", "Z8"])].copy()
plot_series(
    title=f"圖二：Z50 / FC / Z6 / Z7 / Z8 價格 MA{MA_WINDOW}",
    df_sub=df_selected,
    ma_window=MA_WINDOW
)
# ===== 圖三：只顯示 Z61 / Z62 / Z63 =====
df_selected = df[df["camera_model"].isin(["Z6"])].copy()
plot_series(
    title=f"圖三：Z61 / Z62 / Z63 價格 MA{MA_WINDOW}",
    df_sub=df_selected,
    ma_window=MA_WINDOW
)