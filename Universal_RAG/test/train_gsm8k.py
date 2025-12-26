
import sys
import os
# 添加父目录到 path 以便导入 Universal_RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from datasets import load_dataset
import random

def main():
    print("🚀 [GSM8K Train] Starting...")
    
    print("📂 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")['train']
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:200]
    
    training_data = []
    for i in indices:
        training_data.append({
            "question": dataset[i]['question'],
            "ground_truth": dataset[i]['answer']
        })

    model = PrincipleRAGModel(db_path="../math_notebook_db")

    print(f"🧠 Training on {len(training_data)} samples...")
    model.train(training_data)
    print("✅ Training finished!")

if __name__ == "__main__":
    main()
