import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.math_utils import MathEvaluator
from datasets import load_dataset
import random

def main():
    # 1. 准备数据 (MATH Test)
    print("📂 加载 MATH 数据集 (Test Split)...")
    try:
        dataset = load_dataset("jeggers/competition_math", "original", split='test')
    except:
        dataset = load_dataset("lighteval/MATH", "all", split='test')
    
    # 采样 50 条用于演示
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:50]
    
    questions = [dataset[i]['problem'] for i in indices]
    ground_truths = [dataset[i]['solution'] for i in indices]
    dataset_types = [dataset[i]['type'] for i in indices]

    # 2. 初始化模型
    # 使用用户指定的 DB 路径
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 预测
    print("\n🚀 Running Prediction (Baseline 1, Baseline 2, SC-3 + RAG)...")
    # baseline_require=True 会返回 Baseline 1 (SC Fallback) 和 Baseline 2 (Greedy) 的结果
    results = model.predict(questions, baseline_require=True)

    # 4. 评测
    print("\n📈 计算最终统计数据...")
    evaluator = MathEvaluator()
    
    correct_baseline_1 = 0 # SC Fallback
    correct_baseline_2 = 0 # Greedy
    correct_final = 0      # SC + RAG
    
    inconsistent_count = 0
    rag_fixed_count = 0 # Baseline 1 Wrong -> RAG Correct
    
    total = len(questions)
    
    final_results = []
    
    for i, res in enumerate(results):
        gt = ground_truths[i]
        res['ground_truth'] = gt
        res['dataset_type'] = dataset_types[i]
        res['id'] = i
        
        # 1. 验证 Baseline 2 (Greedy)
        is_b2_correct, b2_extracted = evaluator.verify(res.get('baseline_2_raw', ''), gt)
        res['baseline_2_prediction'] = b2_extracted
        res['is_baseline_2_correct'] = is_b2_correct
        if is_b2_correct:
            correct_baseline_2 += 1
            
        # 2. 验证 Baseline 1 (SC Fallback)
        is_b1_correct, b1_extracted = evaluator.verify(res.get('baseline_1_raw', ''), gt)
        res['baseline_1_prediction'] = b1_extracted
        res['is_baseline_1_correct'] = is_b1_correct
        if is_b1_correct:
            correct_baseline_1 += 1
            
        # 3. 验证 Final (SC-3 + RAG)
        is_final_correct, final_extracted = evaluator.verify(res.get('raw_output', ''), gt)
        res['final_prediction'] = final_extracted
        res['is_correct'] = is_final_correct
        if is_final_correct:
            correct_final += 1

        # 4. 统计 RAG 修正情况
        # 如果 RAG 流程被触发 (即 SC 不一致)
        if "RAG" in res.get('method', ''):
            inconsistent_count += 1
            # 只有当 Baseline 1 错误 且 RAG 正确时，才算修正
            if (not is_b1_correct) and is_final_correct:
                rag_fixed_count += 1
                
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
    print(f"Baseline 1 (SC-3 Fallback) Accuracy: {acc_baseline_1:.2f}%")
    print(f"SC-3 + RAG Accuracy: {acc_final:.2f}%")
    print(f"-"*40)
    print(f"Inconsistent Questions (RAG Triggered): {inconsistent_count}")
    print(f"RAG Fixed Wrong Questions: {rag_fixed_count}")
    print(f"RAG Correction Rate (relative to Inconsistent): {ratio_relative_to_inconsistent:.2f}%")
    print(f"RAG Correction Rate (relative to Total): {ratio_relative_to_total:.2f}%")
    print(f"{'='*60}")
    
    OUTPUT_FILE = "math_eval_result.json"
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 处理 retrieved_context 里的对象，转为 dict 或 str
        for r in final_results:
            if 'retrieved_context' in r and r['retrieved_context']:
                # 假设 retrieved_context 是 list of dict
                pass 
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 详细评测日志已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
