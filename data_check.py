import re
import sys
import pandas as pd

INPUT_CSV = "dcview_nikon.csv"
OUTPUT_CSV = "dcview_camera_filtered.csv"

OTHER_BRANDS = [
    "CANON", "SONY", "FUJI", "FUJIFILM", "HASSELBLAD", "LEICA", "PANASONIC", "OLYMPUS",
    "哈蘇", "徠卡", "富士", "索尼", "佳能", "松下", "奧林巴斯"
]

EXCHANGE_WORDS = [
    "貼換", "互換", "交換", "換物", "換機", "以物易物", "TRADE", "SWAP"
]

ACCESSORY_WORDS = [
    "電池", "握把", "充電器", "閃燈", "腳架", "快門線", "記憶卡", "背帶", "皮套",
    "轉接環", "FTZ", "濾鏡", "UV", "CPL", "ND", "防潮箱", "相機包", "保護鏡", "快門數"
]

BUNDLE_WORDS = [
    "含鏡", "帶鏡", "加鏡", "鏡頭套", "套機", "KIT", "雙鏡", "單鏡", "KIT LENS" ,"+", "和"
]

def norm_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("（", "(").replace("）", ")").replace("／", "/").replace("\u00a0", " ")
    s = s.replace("Ⅱ", "II").replace("Ⅲ", "III")
    return s.upper()

def has_lens_pattern(t: str) -> bool:
    if re.search(r"\b\d{1,3}\s*-\s*\d{1,3}\s*MM\b", t):
        return True
    if re.search(r"\b\d{1,3}\s*MM\b", t):
        return True
    if re.search(r"\bF\s*/?\s*\d+(\.\d+)?\b", t):
        return True
    if re.search(r"\b(AF[-\s]?S|AF[-\s]?P|DX|FX|VR|ED)\b", t):
        return True
    if any(w in t for w in ["鏡頭", "定焦", "變焦", "廣角", "望遠", "微距", "魚眼", "增距", "TELECONVERTER", "EXTENDER"]):
        return True
    return False

def has_other_brand(t: str) -> bool:
    return any(b in t for b in OTHER_BRANDS)

def is_exchange(t: str) -> bool:
    return any(w in t for w in EXCHANGE_WORDS)

def is_bundle(t: str) -> bool:
    if any(w in t for w in BUNDLE_WORDS):
        return True
    if "+" in t and has_lens_pattern(t):
        return True
    return False

def is_accessory_only(t: str) -> bool:
    return any(w in t for w in ACCESSORY_WORDS)

def is_body_hint(t: str) -> bool:
    return ("機身" in t) or ("BODY" in t)

def detect_base_model(t: str):
    t = norm_text(t)

    z_map = [
        (r"\bZ\s*50\s*II\b|\bZ50II\b", ("Z50", 2)),
        (r"\bZ\s*50\b|\bZ50\b", ("Z50", 1)),
        (r"\bZ\s*6\s*III\b|\bZ6III\b", ("Z6", 3)),
        (r"\bZ\s*6\s*II\b|\bZ6II\b", ("Z6", 2)),
        (r"\bZ\s*6\b|\bZ6\b", ("Z6", 1)),
        (r"\bZ\s*7\s*II\b|\bZ7II\b", ("Z7", 2)),
        (r"\bZ\s*7\b|\bZ7\b", ("Z7", 1)),
        (r"\bZ\s*5\s*II\b|\bZ5II\b", ("Z5", 2)),
        (r"\bZ\s*5\b|\bZ5\b", ("Z5", 1)),
        (r"\bZ\s*30\b|\bZ30\b", ("Z30", 1)),
        (r"\bZ\s*FC\b|\bZFC\b", ("ZFC", 1)),
        (r"\bZ\s*F\b|\bZF\b", ("ZF", 1)),
        (r"\bZ\s*8\b|\bZ8\b", ("Z8", 1)),
        (r"\bZ\s*9\b|\bZ9\b", ("Z9", 1)),
    ]

    d_map = [
        (r"\bD\s*6\b|\bD6\b", ("D6", 1)),
        (r"\bD\s*5\b|\bD5\b", ("D5", 1)),
        (r"\bD\s*850\b|\bD850\b", ("D850", 1)),
        (r"\bD\s*810\b|\bD810\b", ("D810", 1)),
        (r"\bD\s*800E?\b|\bD800E?\b", ("D800", 1)),
        (r"\bD\s*780\b|\bD780\b", ("D780", 1)),
        (r"\bD\s*750\b|\bD750\b", ("D750", 1)),
        (r"\bD\s*610\b|\bD610\b", ("D610", 1)),
        (r"\bD\s*600\b|\bD600\b", ("D600", 1)),
        (r"\bD\s*500\b|\bD500\b", ("D500", 1)),
        (r"\bD\s*7500\b|\bD7500\b", ("D7500", 1)),
        (r"\bD\s*7200\b|\bD7200\b", ("D7200", 1)),
        (r"\bD\s*7100\b|\bD7100\b", ("D7100", 1)),
        (r"\bD\s*5600\b|\bD5600\b", ("D5600", 1)),
        (r"\bD\s*5500\b|\bD5500\b", ("D5500", 1)),
        (r"\bD\s*5300\b|\bD5300\b", ("D5300", 1)),
        (r"\bD\s*5200\b|\bD5200\b", ("D5200", 1)),
        (r"\bD\s*5100\b|\bD5100\b", ("D5100", 1)),
        (r"\bD\s*3500\b|\bD3500\b", ("D3500", 1)),
        (r"\bD\s*3400\b|\bD3400\b", ("D3400", 1)),
    ]

    for pat, out in z_map:
        if re.search(pat, t):
            return out
    for pat, out in d_map:
        if re.search(pat, t):
            return out
    return (None, None)

def should_keep_row(title: str) -> bool:
    t = norm_text(title)

    if is_exchange(t):
        return False
    if has_other_brand(t):
        return False
    if is_bundle(t):
        return False
    if is_accessory_only(t) and not is_body_hint(t):
        return False

    if has_lens_pattern(t) and not is_body_hint(t):
        return False

    if has_lens_pattern(t) and is_body_hint(t) and any(w in t for w in BUNDLE_WORDS):
        return False

    return True

def main(inp: str, outp: str):
    df = pd.read_csv(inp)

    df = df[df["trade_type"].astype(str).str.lower().eq("sell")].copy()
    df = df[df["price"].notna()].copy()

    df = df[df["price"] > 10000].copy()
    df = df[df["status"].astype(str).str.lower().eq("sold")].copy()

    df["title_norm"] = df["title"].apply(norm_text)
    df["keep"] = df["title"].apply(should_keep_row)

    df = df[df["keep"]].copy()

    df[["camera_model", "generation"]] = df["title_norm"].apply(lambda t: pd.Series(detect_base_model(t)))

    df = df[df["camera_model"].notna()].copy()

    df_out = df[["camera_model", "generation", "price", "region", "date", "title", "link"]].copy()
    df_out.to_csv(outp, index=False, encoding="utf-8-sig")

    print(f"輸入筆數: {len(pd.read_csv(inp))}")
    print(f"輸出筆數: {len(df_out)}")
    print(f"已輸出: {outp}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    outp = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
    main(inp, outp)
