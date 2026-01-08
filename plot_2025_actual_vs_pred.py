# plot_actual_vs_pred_after_2025.py
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================
# 工具函式
# =========================
def infer_family(model_key: str) -> str:
    s = str(model_key).strip().upper()
    s = re.sub(r"\s*GEN\d+\s*$", "", s)  # 移除尾端 GenX
    m = re.match(r"^([A-Z]+)(\d)", s)
    if m:
        return m.group(1) + m.group(2)
    return s

def months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def normalize_model_name(x: str) -> str:
    return str(x).strip()

def parse_models_arg(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def make_model_key(camera_model: str, generation_int) -> str:
    m = str(camera_model).strip()
    # 特例：z63 永遠不帶 Gen（依你原本邏輯）
    if m.lower() == "z63":
        return "z63"
    if pd.isna(generation_int):
        return m
    return f"{m} Gen{int(generation_int)}"

def get_monthly_series(monthly_df: pd.DataFrame, model_key: str) -> pd.Series:
    g = monthly_df[monthly_df["model_key"] == model_key].sort_values("month")
    if g.empty:
        return pd.Series(dtype=float)
    s = g.set_index("month")["price"].copy()
    # 補齊月缺值（線性插補），確保回測/預測連續
    s = s.resample("MS").median().interpolate(limit_direction="both")
    return s


# =========================
# 特徵表建立（訓練用）
# =========================
def build_features_table(
    monthly_df: pd.DataFrame,
    first_seen: dict,
    family_models: dict,
    lags: list[int],
    ma_windows: list[int],
):
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
        for lag in lags:
            g[f"lag_{lag}"] = g["price"].shift(lag)
        for w in ma_windows:
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

    need_cols = [f"lag_{l}" for l in lags] + [f"ma_{w}" for w in ma_windows] + ["target_next_price"]
    out = out.dropna(subset=need_cols).copy()

    # 補缺值（跟你原本策略一致）
    out["newest_model_price"] = out["newest_model_price"].fillna(0.0)
    out["price_gap_to_newest"] = out["price_gap_to_newest"].fillna(0.0)
    out["vol_3m"] = out["vol_3m"].fillna(0.0)

    return out


# =========================
# 單筆特徵（回測/預測用）
# =========================
def build_one_row_features(
    model_key: str,
    cur_month: pd.Timestamp,
    series_hist: pd.Series,
    monthly_df: pd.DataFrame,
    first_seen: dict,
    family_models: dict,
    lags: list[int],
    ma_windows: list[int],
) -> dict:
    fam = infer_family(model_key)
    model_first = first_seen.get(model_key, series_hist.index.min())

    feat = {
        "model_key": model_key,
        "family": fam,
        # month_idx 會在外面填
        "months_since_first_seen": float(months_diff(cur_month, model_first)),
        "ret_1m": float(series_hist.pct_change(1).iloc[-1]) if len(series_hist) >= 2 else 0.0,
        "vol_3m": float(series_hist.pct_change(1).rolling(3).std().iloc[-1]) if len(series_hist) >= 4 else 0.0,
    }

    for lag in lags:
        feat[f"lag_{lag}"] = float(series_hist.iloc[-lag]) if len(series_hist) >= lag else float(series_hist.iloc[-1])

    for w in ma_windows:
        feat[f"ma_{w}"] = float(series_hist.tail(w).mean()) if len(series_hist) >= w else float(series_hist.mean())

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
        newest_price = monthly_df[
            (monthly_df["model_key"] == newest_model) &
            (monthly_df["month"] == cur_month)
        ]["price"]
        newest_price = float(newest_price.iloc[0]) if len(newest_price) else 0.0
        feat["newest_model_price"] = newest_price
        feat["price_gap_to_newest"] = float(series_hist.iloc[-1] - newest_price) if newest_price != 0.0 else 0.0

    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dcview_camera_filtered_Outlier.csv", help="輸入資料 CSV（需含 camera_model/date/price/generation(可選)）")
    ap.add_argument("--models", default="", help="要畫的 model_key，用逗號分隔，例如：'Z6 Gen3,Z9 Gen1,z63'；留空則取資料量最多前 5")
    ap.add_argument("--split", default="2025-01-01", help="從此日期（含）開始顯示『同月預測 vs 同月實際』，格式 YYYY-MM-DD")
    ap.add_argument("--valid_last_n_months", type=int, default=6, help="驗證集最後 N 個月（用於印 MAE，可不影響圖）")
    ap.add_argument("--forecast_months", type=int, default=12, help="未來預測月數")
    ap.add_argument("--ymax", type=float, default=70000, help="Y 軸上限")
    ap.add_argument("--lags", default="1,2,3", help="lag 列表，例如 '1,2,3'")
    ap.add_argument("--ma", default="3,6", help="移動平均窗口列表，例如 '3,6'")
    args = ap.parse_args()

    LAGS = [int(x) for x in args.lags.split(",") if x.strip()]
    MA_WINDOWS = [int(x) for x in args.ma.split(",") if x.strip()]
    SPLIT_DATE = pd.Timestamp(args.split)

    # 1) 讀資料
    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "generation" in df.columns:
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
        df["generation_int"] = df["generation"].round().astype("Int64")
    else:
        df["generation_int"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df = df.dropna(subset=["camera_model", "date", "price"]).copy()
    df["camera_model"] = df["camera_model"].apply(normalize_model_name)
    df["model_key"] = [
        make_model_key(cm, gi) for cm, gi in zip(df["camera_model"], df["generation_int"])
    ]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()  # 月初 timestamp（MS）

    # 2) 月中位數
    monthly = (
        df.groupby(["model_key", "month"], as_index=False)["price"]
          .median()
          .sort_values(["model_key", "month"])
    )
    if monthly.empty:
        raise ValueError("資料處理後 monthly 為空，請檢查 CSV 欄位與內容。")

    monthly["family"] = monthly["model_key"].apply(infer_family)

    # 3) first_seen / family_models
    first_seen = monthly.groupby("model_key")["month"].min().to_dict()
    family_models = monthly.groupby("family")["model_key"].unique().apply(list).to_dict()

    # 4) 建訓練表
    train_df = build_features_table(monthly, first_seen, family_models, LAGS, MA_WINDOWS)

    # 5) 時序切分（僅用來印 MAE，圖的回測會用 SPLIT_DATE）
    max_month = train_df["month"].max()
    cutoff = (max_month - pd.offsets.MonthBegin(args.valid_last_n_months))
    train_part = train_df[train_df["month"] < cutoff].copy()
    valid_part = train_df[train_df["month"] >= cutoff].copy()

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
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
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

    model = Pipeline(steps=[("prep", preprocess), ("reg", reg)])

    X_train = train_part[cat_features + num_features]
    y_train = train_part[target].astype(float)
    model.fit(X_train, y_train)

    if len(valid_part) > 0:
        X_valid = valid_part[cat_features + num_features]
        y_valid = valid_part[target].astype(float)
        pred = model.predict(X_valid)
        mae = np.mean(np.abs(pred - y_valid.values))
        print(f"✅ Validation MAE ≈ {mae:,.0f} TWD (last {args.valid_last_n_months} months)")
    else:
        print("⚠️ 驗證集為空，略過 MAE。")

    # 6) 決定要畫哪些機型
    target_models = parse_models_arg(args.models)
    target_models = ["Z6 Gen2", "Z50 Gen1",]

    if not target_models:
        target_models = monthly["model_key"].value_counts().head(5).index.tolist()
        print("未指定 --models，改用資料量最多前 5：", target_models)

    # 7) month_idx map（用於回測與未來預測）
    all_months_sorted = sorted(monthly["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months_sorted)}
    base_idx = max(month_to_idx.values()) if month_to_idx else 0

    # 8) 畫圖
    plt.figure(figsize=(13, 7))

    for mk in target_models:
        s = get_monthly_series(monthly, mk)
        if len(s) == 0:
            print(f"⚠️ 找不到 {mk} 的資料，略過。")
            continue

        # 確保月都在 month_to_idx（以防 resample 後產生新月份）
        for m in s.index:
            if m not in month_to_idx:
                base_idx += 1
                month_to_idx[m] = base_idx

        last_real_date = s.index.max()

        # --- (A) 回測：2025(含)後，同月預測 vs 同月實際 ---
        backtest_months = s.index[(s.index >= SPLIT_DATE)]
        pred_bt = []
        pred_bt_months = []

        min_hist_need = max(LAGS + MA_WINDOWS) + 1

        for m in backtest_months:
            hist_end = m - pd.offsets.MonthBegin(1)  # m 的前一個月
            hist = s.loc[:hist_end].copy()
            if len(hist) < min_hist_need:
                continue

            feat = build_one_row_features(
                mk, m, hist,
                monthly, first_seen, family_models,
                LAGS, MA_WINDOWS
            )
            feat["month_idx"] = int(month_to_idx.get(m, 0))

            X_one = pd.DataFrame([feat])[cat_features + num_features]
            yhat = float(model.predict(X_one)[0])

            pred_bt.append(yhat)
            pred_bt_months.append(m)

        pred_bt_series = pd.Series(pred_bt, index=pred_bt_months)

        # --- (B) 未來遞迴預測（接在最後實際資料後） ---
        future_months = pd.date_range(
            start=last_real_date + pd.offsets.MonthBegin(1),
            periods=args.forecast_months,
            freq="MS"
        )

        pred_future = []
        series_ext = s.copy()

        # future 的 month_idx 要往後延伸
        future_idx_map = {}
        cur_idx_base = max(month_to_idx.values()) if month_to_idx else 0
        for i, m in enumerate(future_months):
            future_idx_map[m] = cur_idx_base + i + 1

        for m in future_months:
            feat = build_one_row_features(
                mk, m, series_ext,
                monthly, first_seen, family_models,
                LAGS, MA_WINDOWS
            )
            feat["month_idx"] = int(future_idx_map[m])

            X_one = pd.DataFrame([feat])[cat_features + num_features]
            yhat = float(model.predict(X_one)[0])

            pred_future.append(yhat)
            series_ext.loc[m] = yhat

        # --- (C) 作圖規則 ---
        # 實際：全畫（但讀者理解上：2025前只有實際，2025後會多一條預測線）
        line_actual, = plt.plot(s.index, s.values, linewidth=2.2, label=f"{mk} (actual)")
        c = line_actual.get_color()

        # 2025(含)後：同月預測（回測）
        if len(pred_bt_series) > 0:
            plt.plot(pred_bt_series.index, pred_bt_series.values,
                     linestyle="--", linewidth=2.4, color=c,
                     label=f"{mk} (pred vs actual, >= {SPLIT_DATE.strftime('%Y-%m')})")

        # 未來：遞迴預測（用 : 跟回測區隔）
        plt.plot(future_months, pred_future,
                 linestyle=":", linewidth=2.6, color=c,
                 label=f"{mk} (future forecast)")

    plt.title("Actual vs Predicted (Backtest from 2025) + Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (TWD)")
    plt.grid(True)
    plt.ylim(top=args.ymax)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
