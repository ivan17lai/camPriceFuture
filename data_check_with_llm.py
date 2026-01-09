from unittest import result
import gemma3
import csv
import json
import time

model, processor = gemma3.load_model("google/gemma-3-4b-it")

def run_gemma3_single_product_check(data):

    input_title = data['title']
    input_price = data['price']
    input_date = data['date']
    input_user = data['user']
    input_link = data['link']
    input_region = data['region']
    input_status = data['status']
    input_trade_type = data['trade_type']
    
    prompt = """
    請將使用者提供的「所有文字內容」視為【同一個拍賣/銷售品項】進行解析。

    請判斷該內容是否只包含「單一產品」：
    - 若內容包含兩個或以上不同的實體產品（例如：兩顆不同鏡頭、相機 + 鏡頭、A + B 合售），則 is_single_product = false
    - 若僅為單一相機或單一鏡頭，則 is_single_product = true

    請【只回傳一個 JSON 物件】，不可回傳陣列，不可附加任何說明文字。
    JSON 格式【欄位名稱必須完全一致，不可更改】：
    {
    "brand": "string",
    "model": "string",
    "is_single_product": true | false,
    "shotcount": integer | null
    }

    【model 判定規則】
    - 回傳「標準化後的產品型號代碼」
    - 僅回傳型號本身，不包含品牌、不包含多餘描述

        【相機機身格式】
        - Nikon Z 系列：
          Z50, Zfc(注意小寫), Z5, Z6, Z6II, Z6III, Z7, Z7II, Z8, Z9
        - Nikon D 系列（DSLR）：
          D850, D780, D750, D610, D500, D90 等
        - Nikon 底片機：
          F3, FM2, FE2, F100 等

        【鏡頭格式】
        - Z 接環鏡頭：
          Z24-70F4
          Z24-70F2.8
          Z70-200F2.8
          Z50F1.8
        - F 接環鏡頭：
          F50F1.8D
          F24-70F2.8G
          F105F2.8VR
          保留具識別性的後綴（D / G / E / VR / S / Micro）

        【多產品情況】
        - 若包含兩個或以上不同產品：
        - is_single_product = false
        - model 請用英文逗號連接，如：
            "Z6II, Z24-70F4"

        【無法判定】
        - 無法從文字中判定具體型號時，回傳：
        "Unknown"


    【is_single_product 判定規則】
    - true：
    - 僅出現「一個」可獨立販售的實體產品
    - 例如：
        - 一台相機機身
        - 一顆鏡頭
        - 一顆鏡頭 + 非產品性配件（遮光罩、前後蓋、原廠盒）

    - false：
    - 出現「兩個或以上」可獨立販售的實體產品
    - 例如：
        - 兩顆不同鏡頭
        - 相機機身 + 鏡頭
        - 鏡頭 + 鏡頭
        - 不同品牌或不同型號的產品合售
        - A、B、A+B、/、＋、頓號（、）等表示多項產品的描述

    - 若無法確定是否為單一實體產品，請回傳 false（保守判定）

    【shotcount 判定規則】
    - 若內容中有明確提及快門數，請回傳整數數值
    - 若無法判定快門數，請回傳 null
    使用者提供的文字內容如下：
    """ + f"{{{input_title}}}"

    my_messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    output = ""
    for chunk in gemma3.generate_text(my_messages, model=model, processor=processor):
        print(chunk, end="", flush=True)
        output += chunk
    
    # 嘗試解析 JSON
    try:
        output = output.replace("```json", "").replace("```", "")
        output_json = json.loads(output)
        result = {
            # === 判斷 / 分析結果 ===
            "is_single_product": output_json.get("is_single_product", False),
            "model": output_json.get("model", "Unknown"),
            "brand": output_json.get("brand", "Unknown"),
            "shotcount": output_json.get("shotcount", None),

            # === 產品本身資訊 ===
            "title": input_title,
            "price": input_price,
            "trade_type": input_trade_type,

            # === 交易狀態 / 地區 ===
            "status": input_status,
            "region": input_region,

            # === 發布與來源資訊 ===
            "date": input_date,
            "user": input_user,
            "link": input_link
        }

        return result
    except Exception as e:
        print("\n\n無法解析為 JSON，錯誤：", e)
        return None


with open('dcview_nikon.csv', 'r', encoding='utf-8') as f_in, \
    open('dcview_nikon_with_llm.csv', 'w', encoding='utf-8', newline='') as f_out:

    reader = csv.DictReader(f_in)

    fieldnames = [
       "is_single_product",
       "model",
       "brand",
       "shotcount",
       "title",
       "price",
       "trade_type",
       "status",
       "region",
       "date",
       "user",
       "link",
       "success"
    ]

    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:

        start_time = time.time()

        result = None
        success = False

        try:
            result = run_gemma3_single_product_check(row)
            success = True
        except Exception as e:
            print("Error running LLM for row:", e)

        if not result:
            result = {
                "is_single_product": None,
                "model": None,
                "brand": None,
                "shotcount": None,
                "title": row.get("title"),
                "price": row.get("price"),
                "trade_type": row.get("trade_type"),
                "status": row.get("status"),
                "region": row.get("region"),
                "date": row.get("date"),
                "user": row.get("user"),
                "link": row.get("link"),
            }
            success = False 

        sc = result.get("shotcount")
        if isinstance(sc, str) and sc.isdigit():
            result["shotcount"] = int(sc)
        elif not isinstance(sc, int):
            result["shotcount"] = None

        result["success"] = success
        writer.writerow(result)
        f_out.flush()
        print(f"wrote row success={success} model={result.get('model')}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")
        print(f"預計完成時間：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time + (end_time - start_time) * 60000))}\n")