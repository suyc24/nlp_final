
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.math_utils import MathEvaluator
from datasets import load_dataset
import random

def main():
    print("🚀 [MATH Train] Starting...")
    
    # 1. 准备数据 (MATH Train)
    print("📂 Loading MATH dataset...")
    # 注意：MATH 数据集通常较大，且加载可能需要特定配置
    try:
        dataset = load_dataset("jeggers/competition_math", "original", split='train')
    except:
        print("⚠️ Failed to load jeggers/competition_math, trying lighteval/MATH...")
        dataset = load_dataset("lighteval/MATH", "all", split='train')

    # 采样 200 条用于演示
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:200]
    
    training_data = []
    for i in indices:
        training_data.append({
            "question": dataset[i]['problem'],
            "ground_truth": dataset[i]['solution']
        })

    # 2. 初始化模型
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 定义 MATH 专用的验证函数
    evaluator = MathEvaluator()
    
    def math_verifier(pred_text, gt_text):
        # MathEvaluator.verify 返回 (bool, extracted_text)
        is_correct, _ = evaluator.verify(pred_text, gt_text)
        return is_correct

    # 4. 开始训练
    print(f"🧠 Training on {len(training_data)} samples with MathEvaluator...")
    model.train(training_data, verifier_func=math_verifier)
    print("✅ Training finished!")

if __name__ == "__main__":
    main()
