import asyncio
import random
import time
import wandb
import nest_asyncio
from src.utils.dataset_loader import load_dataset_by_name
from src.utils.evaluator import Evaluator
from src.agents.agent_config import create_agents
import os

# 启用异步支持
nest_asyncio.apply()

def main():

    os.environ["http_proxy"] = "http://127.0.0.1:11434"
    os.environ["https_proxy"] = "http://127.0.0.1:11434"

    # 初始化 wandb
    os.environ["WANDB_MODE"] = "offline"
    os.environ['WANDB_INIT_TIMEOUT'] = '600'
    wandb.login(key = "6083d80f5ad186ec2c18d41b80e13d8a0a17d986")
    wandb.init(project="multi-agent-demo")

    print("1")

    # 初始化评估器
    evaluator = Evaluator()

    # 加载数据集
    math_dataset = load_dataset_by_name("gsm8k")

    # 随机选择一个问题
    random_number = random.randint(1, 500)
    sample_question = math_dataset[random_number]["question"]
    sample_answer = math_dataset[random_number]["answer"]

    # 创建智能体
    user, manager, group_chat = create_agents()

    # 开始对话
    user.initiate_chat(
        manager,
        message=sample_question,
        max_turns=5
    )

    # 记录结果
    result = {
        "question": sample_question,
        "answer": group_chat.messages[-1]["content"],
        "steps": [msg["content"] for msg in group_chat.messages]
    }

    # 计算执行时间
    start_time = time.time()
    duration = time.time() - start_time

    # 计算得分
    em_score = evaluator.exact_match(result["answer"], sample_answer)

    # 记录到 wandb
    wandb.log({
        "question": sample_question,
        "exact_match": em_score,
        "time": duration
    })

    print(f"最终答案: {result['answer']}")

if __name__ == "__main__":
    main()