import json
import csv
import os

# 读取JSON文件
logs_directory = r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat"
experiment_name = r"IL_and_Mixed经典PFSP_多技术流派_并行_分层_rule3_0.3_5类-run-20260523-150613"

json_path = os.path.join(logs_directory, experiment_name, 'Elite_Fire_Stats.json')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取value数量（取第一个key的list长度）
num_values = len(next(iter(data.values())))

# 准备CSV数据
rows = []
for actor_name, values in data.items():
    rows.append([actor_name] + values) # 列表拼接

# 按actor名称排序（提取数字部分）
def extract_number(name):
    try:
        return int(name.replace('actor_rein', ''))
    except:
        return 0

rows.sort(key=lambda x: extract_number(x[0]))

# 保存为CSV（同目录）
csv_path = os.path.join(os.path.dirname(json_path), 'Elite_Fire_Stats.csv')

# 动态生成header
header = ['actor_name'] + [f'value_{i+1}' for i in range(num_values)]

with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV已保存至: {csv_path}")
print(f"共转换 {len(rows)} 条记录")

# 显示前10行
print("\n前10行数据:")
for row in rows[:10]:
    values_str = ', '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in row[1:]])
    print(f"  {row[0]}: {values_str}")
