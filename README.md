# UrbanTransit-Assistant 城市轨道交通应急处置助手

基于 **Qwen2.5-7B + LoRA 微调 + Ollama 本地部署** 的地铁应急处置 AI 助手。

## 🎯 项目目标

为地铁运营公司提供一个内部 AI 助手，能够根据公司《地铁突发事件应急预案》，准确回答工作人员关于火灾、信号故障、人流拥挤时的标准处置流程。

## 📁 项目结构

```
metro-emergency-assistant/
├── data/                    # 训练数据
│   ├── raw/                 # 原始预案文档
│   ├── processed/           # 处理后的数据
│   └── train_data.json      # 训练数据集
├── scripts/                 # 脚本文件
│   ├── prepare_data.py      # 数据预处理
│   ├── train_lora.py        # LoRA 微调
│   ├── merge_lora.py        # 合并权重
│   └── convert_to_gguf.py   # 转换 GGUF
├── src/                     # 源代码
│   ├── api.py               # FastAPI 服务
│   └── config.py            # 配置文件
├── models/                  # 模型文件
├── ollama_deploy/           # Ollama 配置
│   └── Modelfile            # 模型定义
└── tests/                   # 测试文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备训练数据

```bash
python scripts/prepare_data.py
```

### 3. LoRA 微调

```bash
python scripts/train_lora.py
```

### 4. 转换并部署到 Ollama

```bash
python scripts/merge_lora.py
python scripts/convert_to_gguf.py
ollama create metro-emergency-assistant -f ollama_deploy/Modelfile
```

### 5. 启动 API 服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 📖 API 使用

### 对话接口

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "地铁站发生火灾时应该怎么处理？"}'
```

### 流式对话

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "信号故障时司机应该怎么处理？"}'
```

## 🔧 技术栈

- **基础模型**: Qwen2.5-7B
- **微调方法**: LoRA (Low-Rank Adaptation)
- **推理框架**: Ollama
- **API 框架**: FastAPI
- **训练框架**: Hugging Face Transformers + PEFT

## 📋 支持的应急场景

| 场景 | 描述 |
|-----|------|
| 🔥 火灾应急 | 发现火情、初期灭火、人员疏散、上报流程 |
| 🚦 信号故障 | 故障识别、降级运营、司机操作规范 |
| 👥 人流拥挤 | 大客流预警、限流措施、疏导方案 |

## 📄 License

MIT License
