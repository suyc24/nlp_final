
import sys
import os
# 添加父目录到 path 以便导入 Universal_RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from datasets import load_dataset
import random

def main():
    print("🚀 [GSM8K Train] Starting...")
    
    # 1. 准备数据 (GSM8K Train)
    print("📂 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")['train']
    
    # 采样 200 条用于演示训练 (实际使用时可加大)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:200]
    
    training_data = []
    for i in indices:
        training_data.append({
            "question": dataset[i]['question'],
            "ground_truth": dataset[i]['answer']
        })

    # 2. 初始化模型
    # 指定数据库路径，以便与 MATH 区分
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 开始训练
    # GSM8K 使用默认的数值验证器，无需传入 verifier_func
    print(f"🧠 Training on {len(training_data)} samples...")
    model.train(training_data)
    print("✅ Training finished!")

if __name__ == "__main__":
    main()
