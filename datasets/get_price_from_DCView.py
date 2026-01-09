import re
import csv
import requests
from bs4 import BeautifulSoup

Brand = "Nikon"
BASE_URL = f"http://market.dcview.com/brand/{Brand}"

MAX_RETRY = 3
SLEEP_SECONDS = 60

START_PAGE = 1  #開始頁數
END_PAGE = 3000   #結束頁數

def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text

def parse_desktop_row(tr):
    tds = tr.find_all("td")
    brand = tds[0].get_text(strip=True)

    title_td = tds[1]
    a_tag = title_td.find("a")
    link = a_tag["href"] if a_tag else None

    btn_texts = [s.get_text(strip=True) for s in title_td.select("span.btn")]
    status = "sold" if "成交" in btn_texts else None
    trade_type = "sell" if "售" in btn_texts else "buy" if "徵" in btn_texts else None

    full_text = title_td.get_text(" ", strip=True)
    region = None
    m = re.search(r"\[(.*?)\]", full_text)
    if m:
        region = m.group(1)

    title = full_text
    for x in ["售", "徵", "成交"]:
        title = title.replace(x, "", 1)
    if region:
        title = title.replace(f"[{region}]", "", 1)
    title = re.sub(r"\s+", " ", title).strip()

    price_text = tds[2].get_text(strip=True)
    price_num = re.sub(r"[^\d]", "", price_text)
    price = int(price_num) if price_num else None

    user_a = tds[3].find("a")
    user = user_a.get_text(strip=True) if user_a else tds[3].get_text(strip=True)

    date = tds[4].get_text(strip=True)

    return {
        "brand": brand,
        "status": status,
        "trade_type": trade_type,
        "region": region,
        "title": title,
        "price": price,
        "user": user,
        "date": date,
        "link": link,
    }

def parse_mobile_row(tr):
    a_tag = tr.find("a", href=True)
    link = a_tag["href"] if a_tag else None

    brand_tag = tr.select_one("p small")
    brand = brand_tag.get_text(strip=True) if brand_tag else None

    btn_texts = [s.get_text(strip=True) for s in tr.select("span.btn")]
    status = "sold" if "成交" in btn_texts else None
    trade_type = "sell" if "售" in btn_texts else "buy" if "徵" in btn_texts else None

    h5 = tr.select_one("span.h5")
    full_title = h5.get_text(" ", strip=True) if h5 else ""
    region = None
    m = re.search(r"\[(.*?)\]", full_title)
    if m:
        region = m.group(1)

    title = full_title
    if region:
        title = title.replace(f"[{region}]", "", 1)
    title = re.sub(r"\s+", " ", title.replace("\xa0", " ")).strip()

    price_tag = tr.select_one("small.price")
    price_text = price_tag.get_text(strip=True) if price_tag else ""
    price_num = re.sub(r"[^\d]", "", price_text)
    price = int(price_num) if price_num else None

    user_a = tr.find("a", href=lambda x: x and "/user/" in x)
    user = user_a.get_text(strip=True) if user_a else None

    date = None
    for small in tr.select("p small"):
        t = small.get_text(" ", strip=True)
        m2 = re.search(r"(\d{4}-\d{2}-\d{2})", t)
        if m2:
            date = m2.group(1)
            break

    return {
        "brand": brand,
        "status": status,
        "trade_type": trade_type,
        "region": region,
        "title": title,
        "price": price,
        "user": user,
        "date": date,
        "link": link,
    }

def extract_page(page):
    #print(f"正在抓取 {page}...")
    html = fetch_html(f"{BASE_URL}?page={page}")
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.select("tbody tr.hidden-xs"):
        rows.append(parse_desktop_row(tr))

    for tr in soup.select("tr.data-list-xs.visible-xs"):
        rows.append(parse_mobile_row(tr))

    return rows

def extract_pages(start_page, end_page):
    all_rows = []
    seen_links = set()

    for page in range(start_page, end_page + 1):
        rows = extract_page(page)
        if not rows:
            break
        for r in rows:
            if r["link"] and r["link"] in seen_links:
                continue
            if r["link"]:
                seen_links.add(r["link"])
            all_rows.append(r)

    return all_rows

def write_csv(rows, filename):
    fieldnames = ["brand", "status", "trade_type", "region", "title", "price", "user", "date", "link"]
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

import time
import time
from tqdm import tqdm

if __name__ == "__main__":

    fieldnames = ["brand", "status", "trade_type", "region", "title", "price", "user", "date", "link"]
    seen_links = set()
    total = 0

    with open("dcview_nikon.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(
            total=END_PAGE - START_PAGE + 1,
            desc="Pages",
            unit="page",
            dynamic_ncols=True
        ) as pbar:

            page = START_PAGE

            while page <= END_PAGE:
                retry = 0
                pbar.set_postfix_str("status=fetching")

                while retry <= MAX_RETRY:
                    try:
                        rows = extract_page(page)

                        if not rows:
                            pbar.set_postfix_str("status=empty → stop")
                            page = END_PAGE + 1
                            break

                        new_rows = []
                        for r in rows:
                            link = r.get("link")
                            if link and link in seen_links:
                                continue
                            if link:
                                seen_links.add(link)
                            new_rows.append(r)

                        if new_rows:
                            writer.writerows(new_rows)
                            f.flush()
                            total += len(new_rows)

                        pbar.set_postfix_str("status=done")
                        page += 1
                        pbar.update(1)
                        break

                    except KeyboardInterrupt:
                        pbar.set_postfix_str("status=interrupted")
                        page = END_PAGE + 1
                        break

                    except Exception as e:
                        retry += 1
                        if retry > MAX_RETRY:
                            pbar.set_postfix_str("status=skipped")
                            page += 1
                            pbar.update(1)
                            break
                        else:
                            pbar.set_postfix_str(f"status=retrying {retry}/{MAX_RETRY}")
                            time.sleep(SLEEP_SECONDS)
                            pbar.set_postfix_str(f"status=sleeping {SLEEP_SECONDS}s")

    print(f"完成，共寫入 {total} 筆")
