import csv
import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========= Matplotlib 中文字體 =========
rcParams["font.family"] = "Microsoft JhengHei"
rcParams["axes.unicode_minus"] = False

model_counter = Counter()

# ========= Nikon 機身白名單 =========
NIKON_MODELS = [
    # ===== Z series（無反）=====
    "Z30",
    "Z50",
    "Z50II",   # 你後面分析腳本有用到，這裡補上
    "Zfc",
    "Zf",
    "Z5",
    "Z6",
    "Z6II",
    "Z6III",
    "Z7",
    "Z7II",
    "Z8",
    "Z9",

    # ===== D series（DSLR 高階 / 專業）=====
    "D1", "D1H", "D1X",
    "D2H", "D2HS", "D2X", "D2XS",
    "D3", "D3S", "D3X",
    "D4", "D4S",
    "D5",
    "D6",

    # ===== D series（全片幅 FX）=====
    "Df",
    "D600",
    "D610",
    "D700",
    "D750",
    "D780",
    "D800",
    "D800E",
    "D810",
    "D810A",
    "D850",

    # ===== D series（APS-C DX 中高階）=====
    "D100",
    "D200",
    "D300",
    "D300S",
    "D500",
    "D7500",

    # ===== D series（APS-C DX 入門）=====
    "D90",
    "D80",
    "D70",
    "D70S",
    "D60",
    "D50",
    "D40",
    "D40X",

    "D5600",
    "D5500",
    "D5300",
    "D5200",
    "D5100",
    "D5000",

    "D3500",
    "D3400",
    "D3300",
    "D3200",
    "D3100",
]

# ========= 機型別名（同一台的不同寫法 → canonical）=========
# 你要的重點：Z63 / Z6-III / Z6 III / z6iii 都要視為同一台（Z6III）
MODEL_ALIASES = {
    # Z6III 常見別名
    "Z63": "Z6III",
    "Z6iii": "Z6III",
    "Z62": "Z6II",
    "Z6ii": "Z6II",
    "Z72": "Z7II",
    "Z7ii": "Z7II",
    "Z52": "Z5",
    "Z5ii": "Z5",
    "Z502": "Z50II",
    "Z50ii": "Z50II",
}

# ========= canonical 顯示（你想輸出到 CSV 的標準名稱）=========
# 內部比對全用大寫無分隔；輸出時把 ZFC/ZF/DF 轉成你習慣的寫法
def canonical_display_from_norm(norm: str) -> str:
    if norm == "ZFC":
        return "Zfc"
    if norm == "ZF":
        return "Zf"
    if norm == "DF":
        return "Df"
    return norm  # 其他機型一律用大寫（Z6II / D850...）

# ========= 將一段字串正規化成比對用 key（移除空白與連字號等）=========
def normalize_token(s: str) -> str:
    s = s.upper()
    # 移除常見分隔符（空白、連字號、底線、全形連字號等）
    s = re.sub(r"[\s\-_－—]+", "", s)
    # 再保險：只留 A-Z0-9
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s

# ========= 建立「所有可辨識 token → canonical」對照 =========
# 1) 白名單本體
TOKEN_TO_CANONICAL = {}
for m in NIKON_MODELS:
    norm = normalize_token(m)
    TOKEN_TO_CANONICAL[norm] = canonical_display_from_norm(norm)

# 2) 別名
for alias, canonical in MODEL_ALIASES.items():
    TOKEN_TO_CANONICAL[normalize_token(alias)] = canonical

# ========= 建立 regex：允許機型中間有空白/連字號 =========
# 例如 Z6III 可匹配：Z6III / Z6 III / Z6-III / z6iii
def token_to_fuzzy_regex(token: str) -> str:
    # token 這裡用「正規化後 token」（只含 A-Z0-9）
    # 允許每個字元中間插入 0~多個空白或連字號
    parts = [re.escape(ch) for ch in token]
    return r"(?:%s)" % r"[\s\-_－—]*".join(parts)

ALL_TOKENS = sorted(TOKEN_TO_CANONICAL.keys(), key=len, reverse=True)
MODEL_PATTERN = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(token_to_fuzzy_regex(t) for t in ALL_TOKENS) + r")(?![A-Z0-9])",
    re.IGNORECASE
)

# ========= kit / 合售 / 非單機（你原本的規則）=========
KIT_KEYWORDS = [
    "kit", "套組", "含鏡", "含 24", "含24", "含 18", "含18",
    "24-70", "24 70", "18-55", "16-50",
    "+", "＋", "、", "/", "搭", "配",
    "鏡頭", "lens", "body+", "Zeiss", "ZF.2"
]
KIT_PATTERN = re.compile("|".join(re.escape(k) for k in KIT_KEYWORDS), re.IGNORECASE)

# ========= 主判定函式 =========
def detect_camera_body(title: str):
    """
    判定是否為「單一 Nikon 機身」
    條件：
    1) 標題中要能辨識到 Nikon 機型（含大小寫/分隔符/別名）
    2) 只能對應到「一個且僅一個」canonical 機型（否則剃除）
    3) 不得包含 kit / 合售 / 鏡頭等關鍵字
    """
    if not title:
        return False, None, None

    text = title.strip()

    # 1) 找出所有機型 token
    matches = [m.group(1) for m in MODEL_PATTERN.finditer(text)]
    if not matches:
        return False, None, None

    # 2) 正規化並映射到 canonical
    canonical_models = []
    for raw in matches:
        norm = normalize_token(raw)
        canonical = TOKEN_TO_CANONICAL.get(norm)
        if canonical:
            canonical_models.append(canonical)

    if not canonical_models:
        return False, None, None

    # 3) 只允許一個 unique 機型（大小寫/別名都已正規化）
    unique_models = set(canonical_models)
    if len(unique_models) != 1:
        return False, None, None

    model = next(iter(unique_models))

    # 4) kit / 合售檢查
    if KIT_PATTERN.search(text):
        return False, None, None

    return True, model, "Nikon"

# ========= CSV 處理 =========
INPUT_PATH = "datasets/dcview_nikon.csv"
OUTPUT_PATH = "datasets/dcview_nikon_body_only.csv"

with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f_out:

    reader = csv.DictReader(f_in)

    fieldnames = [
        "is_camera",
        "model",
        "brand",
        "title",
        "price",
        "trade_type",
        "status",
        "region",
        "date",
        "user",
        "link"
    ]

    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(reader, start=1):
        title = row.get("title", "")

        is_camera, model, brand = detect_camera_body(title)
        if not is_camera:
            continue

        result = {
            "is_camera": is_camera,
            "model": model,
            "brand": brand,
            "title": title,
            "price": row.get("price"),
            "trade_type": row.get("trade_type"),
            "status": row.get("status"),
            "region": row.get("region"),
            "date": row.get("date"),
            "user": row.get("user"),
            "link": row.get("link"),
        }

        writer.writerow(result)
        model_counter[model] += 1

        print(f"[{i}] processed | is_camera={is_camera} | model={model}")

print(f"✅ 完成：已輸出 {OUTPUT_PATH}")

# ========= 機型數量統計圖 =========
if model_counter:
    models, counts = zip(*model_counter.most_common())
    plt.figure(figsize=(12, 6))
    plt.bar(models, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Nikon 單機身型號數量統計")
    plt.xlabel("機型")
    plt.ylabel("數量")
    plt.tight_layout()
    plt.show()
