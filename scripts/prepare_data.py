import json
import os
from pathlib import Path
import random

def validate_data(file_path):
    """验证训练数据的格式"""
    print(f"正在读取数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"错误: JSON 解析失败 - {e}")
            return None

    if not isinstance(data, list):
        print("错误: 数据必须是 JSON 数组")
        return None

    print(f"成功加载 {len(data)} 条数据")
    
    valid_count = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"警告: 第 {idx+1} 条数据不是字典")
            continue
            
        required_keys = ['instruction', 'input', 'output']
        missing_keys = [k for k in required_keys if k not in item]
        
        if missing_keys:
            print(f"警告: 第 {idx+1} 条数据缺少键: {missing_keys}")
            continue
            
        valid_count += 1

    print(f"数据校验完成: {valid_count}/{len(data)} 条数据格式正确")
    return data

def format_data_for_inspection(data, num_examples=3):
    """打印几条示例数据以供人工检查"""
    print("\n--- 数据示例 ---")
    samples = random.sample(data, min(num_examples, len(data)))
    for i, sample in enumerate(samples):
        print(f"\n[示例 {i+1}]")
        print(f"Instruction: {sample['instruction']}")
        if sample.get('input'):
            print(f"Input: {sample['input']}")
        print(f"Output: {sample['output']}")
        print("-" * 50)

if __name__ == "__main__":
    # 使用 config 中定义的路径（需确保在项目根目录运行或调整 pythonpath）
    # 这里直接使用相对路径，假设在项目根目录运行
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "train_data.json"
    
    if not data_path.exists():
        print(f"错误: 未找到数据文件 {data_path}")
        exit(1)
        
    data = validate_data(data_path)
    if data:
        format_data_for_inspection(data)
