
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.math_utils import MathEvaluator
from datasets import load_dataset
import random

def main():
    print("🚀 [MATH Train] Starting...")
    
    print("📂 Loading MATH dataset...")
    try:
        dataset = load_dataset("jeggers/competition_math", "original", split='train')
    except:
        print("⚠️ Failed to load jeggers/competition_math, trying lighteval/MATH...")
        dataset = load_dataset("lighteval/MATH", "all", split='train')

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:200]
    
    training_data = []
    for i in indices:
        training_data.append({
            "question": dataset[i]['problem'],
            "ground_truth": dataset[i]['solution']
        })

    model = PrincipleRAGModel(db_path="../math_notebook_db")

    evaluator = MathEvaluator()
    
    def math_verifier(pred_text, gt_text):
        # MathEvaluator.verify 返回 (bool, extracted_text)
        is_correct, _ = evaluator.verify(pred_text, gt_text)
        return is_correct

    print(f"🧠 Training on {len(training_data)} samples with MathEvaluator...")
    model.train(training_data, verifier_func=math_verifier)
    print("✅ Training finished!")

if __name__ == "__main__":
    main()
