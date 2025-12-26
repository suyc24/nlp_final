import sys
import os
import json
import random
from datasets import load_dataset

# 确保能引用到上一级目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.math_utils import MathEvaluator

def main():
    # 1. 准备数据 (MATH Test)
    print("📂 加载 MATH 数据集 (Test Split)...")
    try:
        # 尝试加载 HuggingFace 的 MATH 数据集
        dataset = load_dataset("jeggers/competition_math", "original", split='test')
    except Exception as e:
        print(f"⚠️ 加载 jeggers/competition_math 失败，尝试 lighteval/MATH... ({e})")
        try:
            dataset = load_dataset("lighteval/MATH", "all", split='test')
        except Exception as e2:
            print(f"❌ 无法加载数据集: {e2}")
            return
    
    # 固定随机种子以便复现
    random.seed(42)
    
    # 采样 200 条用于快速评测 (如果想跑全量，注释掉切片)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:200]
    
    # 兼容不同的数据集列名 (problem/question, solution/answer)
    questions = []
    ground_truths = []
    dataset_types = []
    
    for i in indices:
        item = dataset[i]
        questions.append(item.get('problem') or item.get('question'))
        ground_truths.append(item.get('solution') or item.get('answer'))
        dataset_types.append(item.get('type', 'Unknown'))

    # 2. 初始化模型
    print("🤖 初始化模型...")
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 预测
    print(f"\n🚀 开始预测 {len(questions)} 道题目 (Baseline 1, 2, 3 + Final)...")
    results = model.predict(questions, baseline_require=True)

    # 4. 评测
    print("\n📈 计算统计数据 (使用 MathEvaluator)...")
    evaluator = MathEvaluator()
    
    cnt_b1 = 0
    cnt_b2 = 0
    cnt_b3 = 0
    cnt_final = 0
    
    inconsistent_total = 0
    
    # 专门统计“不一致”子集里的表现
    inc_rag_correct = 0     # Final (RAG)
    inc_sc_correct = 0      # Baseline 2 (SC Majority)
    inc_greedy_correct = 0  # Baseline 3 (Greedy)
    
    total = len(questions)
    final_results = []
    
    for i, res in enumerate(results):
        gt = ground_truths[i]
        
        # 使用 MathEvaluator 进行特殊的数学等价性验证
        # verify 返回 (bool, extracted_answer)
        b1_ok, _ = evaluator.verify(res.get('baseline_1', ''), gt)
        b2_ok, _ = evaluator.verify(res.get('baseline_2', ''), gt)
        b3_ok, _ = evaluator.verify(res.get('baseline_3', ''), gt)
        final_ok, _ = evaluator.verify(res.get('final_answer', ''), gt)
        
        if b1_ok: cnt_b1 += 1
        if b2_ok: cnt_b2 += 1
        if b3_ok: cnt_b3 += 1
        if final_ok: cnt_final += 1

        # 深入分析不一致的情况
        if not res.get("is_consistent", True):
            inconsistent_total += 1
            if final_ok: inc_rag_correct += 1
            if b2_ok: inc_sc_correct += 1
            if b3_ok: inc_greedy_correct += 1
        
        # 补充信息用于保存
        res['ground_truth'] = gt
        res['dataset_type'] = dataset_types[i]
        res['is_correct'] = final_ok
        res['is_baseline_3_correct'] = b3_ok
        final_results.append(res)

    # Calculate global accuracy
    acc_b1 = cnt_b1 / total * 100
    acc_b2 = cnt_b2 / total * 100
    acc_b3 = cnt_b3 / total * 100
    acc_final = cnt_final / total * 100
    
    # Calculate accuracy within the Inconsistent subset
    acc_inc_rag = (inc_rag_correct / inconsistent_total * 100) if inconsistent_total else 0
    acc_inc_sc = (inc_sc_correct / inconsistent_total * 100) if inconsistent_total else 0
    acc_inc_greedy = (inc_greedy_correct / inconsistent_total * 100) if inconsistent_total else 0
    
    print(f"\n{'='*20} Evaluation Results (MATH) {'='*20}")
    print(f"Total Questions: {total}")
    print(f"Baseline 1 (Direct Greedy):              {acc_b1:.2f}%")
    print(f"Baseline 2 (SC Majority Vote):           {acc_b2:.2f}%")
    print(f"Baseline 3 (SC Consistent + Greedy):     {acc_b3:.2f}%")
    print(f"Final      (SC Consistent + RAG):        {acc_final:.2f}%")
    print(f"-"*40)
    print(f"Inconsistent Questions (subset size):    {inconsistent_total}")
    print(f"--- Battle in the Inconsistent Set ---")
    print(f"  [Baseline 2] SC Majority Acc:          {acc_inc_sc:.2f}%")
    print(f"  [Baseline 3] Greedy Acc:               {acc_inc_greedy:.2f}%")
    print(f"  [Final]      RAG Acc:                  {acc_inc_rag:.2f}%")
    print(f"-"*40)
    
    # Conclusion analysis
    diff_rag_greedy = acc_final - acc_b3

    OUTPUT_FILE = "math_eval_result.json"
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 Detailed evaluation log saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()