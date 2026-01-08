import pandas as pd

INPUT_CSV = "dcview_camera_filtered.csv"
OUTPUT_FILTERED_CSV = "dcview_camera_filtered_Outlier.csv"  # 依你要求：覆寫
OUTPUT_OUTLIER_CSV = "dcview_camera_filtered_Outlier2.csv"

IQR_K = 1.5          # 1.5 是常用標準
MIN_GROUP_SIZE = 6   # 每組資料筆數太少時不做離群值判定，避免誤殺

df = pd.read_csv(INPUT_CSV)

# 基本型別整理
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 去掉必要欄位缺失
df = df.dropna(subset=["camera_model", "generation", "price"]).copy()

# generation 若是 2.0 / 3.0 這種，轉成 int
df["generation"] = df["generation"].astype(int)

def split_outliers_iqr(group: pd.DataFrame) -> pd.DataFrame:
    """回傳 group 並加上 outlier_flag 欄位"""
    g = group.copy()

    # 資料太少：不做離群值判斷，全部視為非離群
    if len(g) < MIN_GROUP_SIZE:
        g["outlier_flag"] = False
        return g

    q1 = g["price"].quantile(0.25)
    q3 = g["price"].quantile(0.75)
    iqr = q3 - q1

    # iqr=0 代表價格幾乎都一樣，也不判斷離群（避免全被判掉）
    if iqr == 0:
        g["outlier_flag"] = False
        return g

    lower = q1 - IQR_K * iqr
    upper = q3 + IQR_K * iqr

    g["outlier_flag"] = (g["price"] < lower) | (g["price"] > upper)
    return g

# 依機型 + 世代分組做離群值標記
df_flagged = (
    df.groupby(["camera_model", "generation"], group_keys=False)
      .apply(split_outliers_iqr)
)

# 移除 camera_model 欄位以 D 或 d 開頭的機型（只保留不以 D/d 開頭的資料）
mask = df_flagged["camera_model"].astype(str).str.startswith(("D", "d"))
removed = mask.sum()
if removed:
    print(f"已移除 D 開頭機型筆數：{removed}")
df_flagged = df_flagged[~mask].copy()

df_outliers = df_flagged[df_flagged["outlier_flag"]].drop(columns=["outlier_flag"])
df_filtered = df_flagged[~df_flagged["outlier_flag"]].drop(columns=["outlier_flag"])

# 輸出
df_filtered.to_csv(OUTPUT_FILTERED_CSV, index=True, encoding="utf-8-sig")
df_outliers.to_csv(OUTPUT_OUTLIER_CSV, index=False, encoding="utf-8-sig")

print(f"原始筆數：{len(df)}")
print(f"保留筆數：{len(df_filtered)}")
print(f"離群值筆數：{len(df_outliers)}")
print(f"已輸出：{OUTPUT_OUTLIER_CSV}")
