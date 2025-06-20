from datasets import load_dataset

def load_dataset_by_name(name):
    """加载指定数据集"""
    if name == "gsm8k":
        return load_dataset("gsm8k", "main")["test"]
    elif name == "hotpotqa":
        return load_dataset("hotpot_qa", "distractor")['validation']
    elif name == "humaneval":
        return load_dataset("openai_humaneval")["test"]
    else:
        raise ValueError(f"未知数据集: {name}") 