import os
import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import MERGED_MODEL_DIR

def convert():
    print("--- GGUF 模型转换指南 ---")
    print("由于 GGUF 转换依赖 llama.cpp 工具，请按照以下步骤操作：")
    
    print(f"\n1. 确保您的合并模型已保存至: {MERGED_MODEL_DIR}")
    
    print("\n2. 克隆 llama.cpp 仓库:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   cd llama.cpp")
    print("   pip install -r requirements.txt")
    
    print(f"\n3. 执行转换命令 (将模型转换为 fp16 GGUF):")
    print(f"   python convert.py {MERGED_MODEL_DIR} --outtype f16 --outfile {MERGED_MODEL_DIR}/metro-assistant-fp16.gguf")
    
    print(f"\n4. (可选) 量化为 4-bit (推荐用于部署):")
    print(f"   ./quantize {MERGED_MODEL_DIR}/metro-assistant-fp16.gguf {MERGED_MODEL_DIR}/metro-assistant-q4_k_m.gguf q4_k_m")
    
    print("\n5. 更新 Ollama Modelfile:")
    print("   修改 ollama_deploy/Modelfile 中的 FROM 路径指向生成的 .gguf 文件")
    
    print("\n注意：如果您已在本地安装了 llama.cpp，您可以直接运行上述命令。")

if __name__ == "__main__":
    convert()
