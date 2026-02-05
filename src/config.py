"""
UrbanTransit-Assistant 配置文件
城市轨道交通应急处置助手
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = DATA_DIR / "train_data.json"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
LORA_WEIGHTS_DIR = MODELS_DIR / "lora_weights"
MERGED_MODEL_DIR = MODELS_DIR / "merged"

# Ollama 配置目录
OLLAMA_DIR = PROJECT_ROOT / "ollama_deploy"


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量化配置 (QLoRA)
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = False


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    
    # 优化器
    optim: str = "paged_adamw_32bit"
    
    # 精度
    fp16: bool = False
    bf16: bool = True
    
    # 日志和保存
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # 输出目录
    output_dir: str = str(LORA_WEIGHTS_DIR)


@dataclass
class OllamaConfig:
    """Ollama 配置"""
    model_name: str = "metro-emergency-assistant"
    base_url: str = "http://localhost:11434"
    
    # 推理参数
    temperature: float = 0.3
    top_p: float = 0.9
    num_ctx: int = 4096
    
    # 系统提示词
    system_prompt: str = """你是城市轨道交通应急处置助手，专门为地铁运营工作人员提供《地铁突发事件应急预案》相关的咨询服务。

你的职责：
1. 准确回答关于火灾、信号故障、人流拥挤等突发事件的标准处置流程
2. 提供具体的硬性规定，包括上报时限、联系部门、操作步骤
3. 回答必须专业、准确，严格按照预案规定

注意事项：
- 如果遇到不在预案范围内的问题，请明确告知用户
- 涉及人员安全的问题，始终优先考虑人员疏散和安全"""


@dataclass
class APIConfig:
    """API 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS 配置
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


# 默认配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
ollama_config = OllamaConfig()
api_config = APIConfig()
