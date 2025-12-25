
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.utils import default_check_correct
from datasets import load_dataset
import random

def main():
    # 1. 准备数据 (GSM8K Test)
    print("📂 加载测试集 (GSM8K Test)...")
    dataset = load_dataset("gsm8k", "main")['test']
    
    # 采样 50 条用于演示评测
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:50]
    
    questions = [dataset[i]['question'] for i in indices]
    ground_truths = [dataset[i]['answer'] for i in indices]

    # 2. 初始化模型 (加载训练好的数据库)
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 预测
    print("\n🚀 Running Prediction (Baseline 1, Baseline 2, SC-3 + RAG)...")
    # baseline_require=True 会返回 Baseline 1 (SC Majority Vote) 和 Baseline 2 (Greedy) 的结果
    results = model.predict(questions, baseline_require=True)

    # 4. 评测
    print("\n📈 计算最终统计数据...")
    
    correct_baseline_1 = 0 # SC Majority Vote (即使不一致也取众数)
    correct_baseline_2 = 0 # Greedy
    correct_final = 0      # SC + RAG
    
    inconsistent_count = 0
    rag_fixed_count = 0 # Baseline 1 Wrong -> RAG Correct
    
    total = len(questions)
    
    final_results = []

    for i, res in enumerate(results):
        gt = ground_truths[i]
        
        # 验证 Baseline 2 (Greedy)
        if 'baseline_2_raw' in res and default_check_correct(res['baseline_2_raw'], gt):
            correct_baseline_2 += 1
            
        # 验证 Baseline 1 (SC Majority Vote - 即使不一致也取众数)
        is_b1_correct = False
        if 'baseline_1_raw' in res:
            is_b1_correct = default_check_correct(res['baseline_1_raw'], gt)
            if is_b1_correct:
                correct_baseline_1 += 1
            
        # 验证 Final (SC-3 + RAG)
        is_final_correct = default_check_correct(res['raw_output'], gt)
        if is_final_correct:
            correct_final += 1
            
        # 统计 RAG 修正情况
        # 如果 RAG 流程被触发 (即 SC 不一致)
        if "RAG" in res['method']:
            inconsistent_count += 1
            # 只有当 Baseline 1 错误 且 RAG 正确时，才算修正
            if (not is_b1_correct) and is_final_correct:
                rag_fixed_count += 1
        
        # 补充信息用于保存
        res['ground_truth'] = gt
        res['id'] = i
        res['is_correct'] = is_final_correct
        final_results.append(res)

    acc_baseline_1 = correct_baseline_1 / total * 100
    acc_baseline_2 = correct_baseline_2 / total * 100
    acc_final = correct_final / total * 100
    
    # 相对于不consistent的比值
    ratio_relative_to_inconsistent = (rag_fixed_count / inconsistent_count * 100) if inconsistent_count > 0 else 0.0
    
    # 相对于总题目数的比值
    ratio_relative_to_total = (rag_fixed_count / total * 100)
    
    print(f"\n{'='*20} Evaluation Results {'='*20}")
    print(f"Total Questions: {total}")
    print(f"Baseline 2 (Greedy) Accuracy: {acc_baseline_2:.2f}%")
    print(f"Baseline 1 (SC-3 Majority Vote) Accuracy: {acc_baseline_1:.2f}%")
    print(f"SC-3 + RAG Accuracy: {acc_final:.2f}%")
    print(f"-"*40)
    print(f"Inconsistent Questions (RAG Triggered): {inconsistent_count}")
    print(f"RAG Fixed Wrong Questions: {rag_fixed_count}")
    print(f"RAG Correction Rate (relative to Inconsistent): {ratio_relative_to_inconsistent:.2f}%")
    print(f"RAG Correction Rate (relative to Total): {ratio_relative_to_total:.2f}%")
    print(f"{'='*60}")
    
    OUTPUT_FILE = "gsm8k_eval_result.json"
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Use default=str to handle non-serializable objects if any
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
