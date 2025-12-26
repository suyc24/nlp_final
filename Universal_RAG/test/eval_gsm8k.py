import sys
import os
import json
import random
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.utils import default_check_correct, extract_answer

def main():
    # 1. 准备数据
    print("📂 加载测试集 (GSM8K Test)...")
    try:
        dataset = load_dataset("gsm8k", "main")['test']
    except:
        print("⚠️ 无法加载 HuggingFace 数据集，尝试本地加载或退出。")
        return

    # 固定随机种子
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:200]
    
    questions = [dataset[i]['question'] for i in indices]
    ground_truths = [dataset[i]['answer'] for i in indices]

    # 2. 初始化模型
    print("🤖 初始化模型...")
    # ⚠️ 请确保这个路径下真的有 math_notebook_db/index.faiss 和 math_notebook_db/meta.json
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. 预测
    print(f"\n🚀 开始预测 {len(questions)} 道题目...")
    results = model.predict(questions, baseline_require=True)

    # 4. 评测统计
    print("\n📈 计算统计数据...")
    
    cnt_b1, cnt_b2, cnt_b3, cnt_final = 0, 0, 0, 0
    inconsistent_total = 0
    inc_rag_correct, inc_sc_correct, inc_greedy_correct = 0, 0, 0
    
    total = len(questions)
    final_results = []

    for i, res in enumerate(results):
        gt_raw = ground_truths[i]
        gt = extract_answer(gt_raw)
        
        # 结果比对
        b1_ok = default_check_correct(res.get('baseline_1', ''), gt)
        b2_ok = default_check_correct(res.get('baseline_2', ''), gt)
        b3_ok = default_check_correct(res.get('baseline_3', ''), gt)
        final_ok = default_check_correct(res.get('final_answer', ''), gt)
        
        if b1_ok: cnt_b1 += 1
        if b2_ok: cnt_b2 += 1
        if b3_ok: cnt_b3 += 1
        if final_ok: cnt_final += 1
            
        # 不一致子集统计
        if not res.get("is_consistent", True):
            inconsistent_total += 1
            if final_ok: inc_rag_correct += 1
            if b2_ok: inc_sc_correct += 1
            if b3_ok: inc_greedy_correct += 1
        
        res['ground_truth'] = gt
        res['is_correct'] = final_ok
        final_results.append(res)

    # 准确率计算
    acc_b1 = cnt_b1 / total * 100
    acc_b2 = cnt_b2 / total * 100
    acc_b3 = cnt_b3 / total * 100
    acc_final = cnt_final / total * 100
    
    acc_inc_rag = (inc_rag_correct / inconsistent_total * 100) if inconsistent_total else 0
    acc_inc_sc = (inc_sc_correct / inconsistent_total * 100) if inconsistent_total else 0
    acc_inc_greedy = (inc_greedy_correct / inconsistent_total * 100) if inconsistent_total else 0

    print(f"\n{'='*20} Evaluation Results {'='*20}")
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
    print(f"{'='*60}")
    
    # 简单的 Debug 提示
    if acc_inc_rag == acc_inc_sc:
        print("⚠️ 警告: RAG Accuracy 与 SC Accuracy 完全一致。")
        print("   可能原因: 1. 检索结果为空。 2. LLM 忽略了 Context。 3. 代码未正确传入 Prompt。")
        print("   请检查上方日志中的 '检索统计'。")
    
    output_file = "gsm8k_eval_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 结果已保存至: {output_file}")

if __name__ == "__main__":
    main()