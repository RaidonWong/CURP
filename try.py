# txt_to_json_ids.py

import json

input_file = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/ids.txt"
output_file = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/ids.json"


with open(input_file, "r", encoding="utf-8") as f:
    ids = [line.strip() for line in f if line.strip()]  # 去除空行和换行符


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)

print(f"Transfer Finish! In total {len(ids)}  ID, saved to {output_file}")