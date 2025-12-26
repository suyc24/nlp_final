import sys
import os
import json
import random
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Universal_RAG.core import PrincipleRAGModel
from Universal_RAG.utils import default_check_correct, extract_answer

def main():
    # 1. Prepare data
    print("📂 Loading test set (GSM8K Test)...")
    try:
        dataset = load_dataset("gsm8k", "main")['test']
    except:
        print("⚠️ Unable to load HuggingFace dataset, trying local load or exit.")
        return

    # Fix random seed
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:200]
    
    questions = [dataset[i]['question'] for i in indices]
    ground_truths = [dataset[i]['answer'] for i in indices]

    # 2. Initialize model
    print("🤖 Initializing model...")
    # ⚠️ Please make sure there is math_notebook_db/index.faiss and math_notebook_db/meta.json in this path
    model = PrincipleRAGModel(db_path="../math_notebook_db")

    # 3. Prediction
    print(f"\n🚀 Start predicting {len(questions)} questions...")
    results = model.predict(questions, baseline_require=True)

    # 4. Evaluation statistics
    print("\n📈 Calculating statistics...")
    
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
            
        # Inconsistent subset statistics
        if not res.get("is_consistent", True):
            inconsistent_total += 1
            if final_ok: inc_rag_correct += 1
            if b2_ok: inc_sc_correct += 1
            if b3_ok: inc_greedy_correct += 1
        
        res['ground_truth'] = gt
        res['is_correct'] = final_ok
        final_results.append(res)

    # Accuracy calculation
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
    
    # Simple debug tip
    if acc_inc_rag == acc_inc_sc:
        print("⚠️ Warning: RAG Accuracy is exactly the same as SC Accuracy.")
        print("   Possible reasons: 1. Retrieval results are empty. 2. LLM ignored the Context. 3. Prompt not passed correctly.")
        print("   Please check the 'retrieval statistics' in the log above.")
    
    output_file = "gsm8k_eval_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 Results saved to: {output_file}")

if __name__ == "__main__":
    main()