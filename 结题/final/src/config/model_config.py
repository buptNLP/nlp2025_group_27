# Model configurations for different agents

import os
from dotenv import load_dotenv
import httpx

# 加载环境变量
load_dotenv()

# 1. Planner: Phi-3:mini
phi3_config_list = [
    {
        "model": "phi3:mini",
        "api_type": "ollama",
        "client_host": "http://localhost:11434",
    }
]

# 2. Executor_01: DeepSeek-chat
deepseek_config_list = {
    "model": "deepseek-chat",
    "api_key": "sk-e685cc8450534eba8a195f44de737be8",
    "base_url": "https://api.deepseek.com",
    "api_type": "openai",
}

# 3. Checker: TinyLlama 1.1b
tinyllama_config_list = [
    {
        "model": "tinyllama:1.1b",
        "api_type": "ollama",
        "client_host": "http://localhost:11434",
    }
]

# 4. Reflector: Qwen3-1.7b
qwen_config_list = {
    "model": "qwen3:1.7b",
    "api_type": "ollama",
    "client_host": "http://localhost:11434",
}

# 5. Executor_02: gemma3:1b
gemma_config_list = {
    "model": "gemma3:1b",
    "api_type": "ollama",
    "client_host": "http://localhost:11434",
}

# 6. Executor_03: orca_mini:3b
orca_config_list = {
    "model": "orca-mini:3b",
    "api_type": "ollama",
    "client_host": "http://localhost:11434",
}

def check_ollama_models():
    """检查所需的 Ollama 模型是否已安装"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            installed_models = [model["name"] for model in response.json()["models"]]
            required_models = [
                "phi3:mini",
                "tinyllama:1.1b",
                "qwen3:1.7b",
                "gemma3:1b",
                "orca_mini:3b",
                "llama2"  # 作为后备模型
            ]
            missing_models = [model for model in required_models if model not in installed_models]
            if missing_models:
                print("警告：以下模型未安装：")
                for model in missing_models:
                    print(f"- {model}")
                print("\n请运行以下命令安装缺失的模型：")
                for model in missing_models:
                    print(f"ollama pull {model}")
            return True
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到 Ollama 服务")
        print("请确保 Ollama 服务已启动：ollama serve")
        return False 