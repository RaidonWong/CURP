# txt_to_json_ids.py

import json

input_file = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/ids.txt"
output_file = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/ids.json"

# 读取 txt 文件，每行作为一个 id
with open(input_file, "r", encoding="utf-8") as f:
    ids = [line.strip() for line in f if line.strip()]  # 去除空行和换行符

# 保存为 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成！共 {len(ids)} 个 ID，已保存到 {output_file}")