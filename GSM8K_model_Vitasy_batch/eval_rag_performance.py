import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
import torch
import shutil
import numpy as np
import chromadb
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
import config

# ================= 配置区域 =================
MODEL_PATH = config.EVAL_MODEL_PATH
DB_PATH = config.EVAL_DB_PATH
OUTPUT_FILE = config.OUTPUT_FILE

# RAG 配置
TOP_K = config.TOP_K
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD
BATCH_SIZE_EMBED = config.BATCH_SIZE_EMBED
SC_N = config.SC_N  # Self-Consistency 采样次数

# ================= 1. 检索器 (CPU Mode) =================
class KnowledgeRetriever:
    def __init__(self, db_path):
        print(f"📚 加载知识库: {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="elite_strategies")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        
    def batch_search(self, queries):
        # 如果没有查询，直接返回
        if not queries: return []
        
        print(f"🔍 正在检索 {len(queries)} 条抽象化查询...")
        q_embeddings = self.embedder.encode(
            queries, 
            batch_size=BATCH_SIZE_EMBED, 
            show_progress_bar=True, 
            convert_to_numpy=True
        ).tolist()
        
        search_results = self.collection.query(
            query_embeddings=q_embeddings,
            n_results=TOP_K
        )
        
        retrieved_contexts = []
        for i in range(len(queries)):
            valid_hints = []
            if search_results['ids'][i]:
                for j in range(len(search_results['ids'][i])):
                    distance = search_results['distances'][i][j]
                    doc_text = search_results['documents'][i][j]
                    metadata = search_results['metadatas'][i][j]
                    if distance < SIMILARITY_THRESHOLD:
                        valid_hints.append({
                            "strategy": doc_text,
                            "trigger": metadata.get('trigger', 'Unknown'),
                            "score": 1 - distance
                        })
            retrieved_contexts.append(valid_hints)
        return retrieved_contexts

# ================= 2. 评测引擎 (Adaptive SC) =================
class AdaptiveEvaluator:
    def __init__(self):
        print(f"🚀 初始化评测引擎: {MODEL_PATH}")
        self.llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True
        )
        
        # SC 采样参数: Temperature > 0 才能有多样性
        self.params_sc = SamplingParams(
            n=SC_N,
            temperature=0.7, 
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # RAG 采样参数: T=0 求稳
        self.params_rag = SamplingParams(
            temperature=0.2, 
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # 抽象化参数
        self.params_abs = SamplingParams(temperature=0.0, max_tokens=128)

    # --- Prompt 构造区 ---
    def construct_abstraction_prompt(self, q):
        return f"""<|im_start|>user
Task: Extract the underlying "Math Pattern" from the problem.
1. Remove specific numbers (replace with X, Y, etc.).
2. Remove entity names (e.g., "John" -> "Person", "Apples" -> "Items").
3. Describe the logical structure concisely.

[Example]
Input: John buys 5 apples for $2 each. Total?
Pattern: Calculating total cost given quantity and unit price.

[Target]
Input: {q}
Pattern:<|im_end|>
<|im_start|>assistant
"""

    def construct_base_prompt(self, q):
        return f"<|im_start|>user\nQuestion: {q}\nPlease reason step-by-step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"

    def construct_rag_prompt(self, q, hints):
        if not hints: return self.construct_base_prompt(q)
        
        strategies_text = ""
        for idx, h in enumerate(hints):
            strategies_text += f"Strategy {idx+1} (Matched Scenario: {h['trigger']}):\n{h['strategy']}\n\n"
        
        # --- 修改开始：加入了 [Demonstration] 部分 ---
        content = f"""Reference Knowledge:
{strategies_text}
---
[Demonstration of how to use the Strategy]
Example Scenario:
Reference Strategy: "To find the total distance, multiply the speed by the time."
Question: A car travels at 60 mph for 3 hours. How far does it go?
Reasoning: The Reference Strategy suggests multiplying speed by time. 
Speed = 60, Time = 3. 
Calculation: 60 * 3 = 180.
The answer is \\boxed{{180}}.

---
[Your Turn]
Question: {q}
Instruction: First, check if any of the "Reference Knowledge" above applies to this question. If yes, explicitly use that logic. Reason step-by-step, and put your final answer within \\boxed{{}}."""
        # --- 修改结束 ---

        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    # --- 工具区 ---
    def extract_answer(self, text):
        if not text: return None
        text = text.replace(',', '')
        match = re.search(r'\\boxed\{(\-?\d+\.?\d*)\}', text)
        if match: return float(match.group(1))
        matches = re.findall(r'-?\d+\.?\d*', text[-100:])
        if matches: return float(matches[-1])
        return None

    def check_correct(self, pred, gt):
        if "####" in gt: gold = self.extract_answer(gt.split("####")[1])
        else: gold = self.extract_answer(gt)
        val = self.extract_answer(pred)
        if gold is None or val is None: return False
        return abs(gold - val) < 1e-4

    def get_majority_vote(self, outputs_list):
        """
        输入: 一个 RequestOutput 对象 (包含 n 个 completion)
        输出: (majority_answer, is_consistent)
        is_consistent = True 表示有 >=2 个答案一致
        """
        answers = []
        for output in outputs_list.outputs:
            val = self.extract_answer(output.text)
            if val is not None:
                answers.append(val)
        
        if not answers: return None, False
        
        counts = Counter(answers)
        most_common_val, count = counts.most_common(1)[0]
        
        # SC=3 时，如果有2个或3个一样，就算一致；全是1个，则不一致
        is_consistent = (count >= 2)
        return most_common_val, is_consistent

    # --- 主流程 ---
    def run_evaluation(self):
        retriever = KnowledgeRetriever(DB_PATH)
        print("📂 加载测试集 (GSM8K Test)...")
        dataset = load_dataset("gsm8k", "main")['test']
        questions = dataset['question']
        ground_truths = dataset['answer']
        
        # ==========================================
        # Stage 1: 第一次尝试 - Self-Consistency (SC=3)
        # ==========================================
        print(f"\n⚡️ [Phase 1] 运行 SC-3 投票 (N={len(questions)})...")
        prompts_base = [self.construct_base_prompt(q) for q in questions]
        
        # 这里一次生成 n=3 个候选
        outputs_sc = self.llm.generate(prompts_base, self.params_sc)
        
        # 分析 SC 结果，筛选出需要 RAG 的题目
        rag_indices = [] # 需要 RAG 的题目索引
        rag_questions = [] # 对应的文本
        final_results = [None] * len(questions) # 预占位
        
        consistent_count = 0
        baseline_correct = 0 # 新增：统计 Baseline 正确数
        
        for i, output in enumerate(outputs_sc):
            # 投票
            maj_ans, is_consistent = self.get_majority_vote(output)
            
            # --- Baseline 统计 ---
            gt_text = ground_truths[i]
            if "####" in gt_text: gold = self.extract_answer(gt_text.split("####")[1])
            else: gold = self.extract_answer(gt_text)
            
            if maj_ans is not None and gold is not None and abs(maj_ans - gold) < 1e-4:
                baseline_correct += 1
                if not is_consistent:
                    print(f"\n🎲 [Lucky Guess] ID: {i}")
                    print(f"Question: {questions[i]}")
                    print(f"Ground Truth: {gold}")
                    extracted_vals = [self.extract_answer(o.text) for o in output.outputs]
                    print(f"AI Answers (Extracted): {extracted_vals}")
            # --------------------

            if is_consistent:
                # 投票成功，直接采纳
                consistent_count += 1
                final_results[i] = {
                    "id": i,
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "method": "SC-3 (Consistent)",
                    "prediction": maj_ans,
                    "raw_output": output.outputs[0].text # 存第一个作为参考
                }
            else:
                # 投票失败（3个答案都不一样），进入 RAG 队列
                rag_indices.append(i)
                rag_questions.append(questions[i])
        
        print(f"   -> 一致性通过: {consistent_count}/{len(questions)}")
        print(f"   -> 需要 RAG 介入: {len(rag_questions)}/{len(questions)}")

        # ==========================================
        # Stage 2: 针对不一致题目 - Adaptive RAG
        # ==========================================
        if rag_questions:
            print(f"\n🌀 [Phase 2.1] 对 {len(rag_questions)} 道难题进行抽象化...")
            # 2.1 抽象化
            abs_prompts = [self.construct_abstraction_prompt(q) for q in rag_questions]
            abs_outputs = self.llm.generate(abs_prompts, self.params_abs)
            abstract_queries = [o.outputs[0].text.strip() for o in abs_outputs]
            
            # 2.2 检索
            print(f"\n🔍 [Phase 2.2] 检索经验...")
            hints_list = retriever.batch_search(abstract_queries)
            
            # 2.3 RAG 推理
            print(f"\n⚡️ [Phase 2.3] 运行 RAG 推理 (Greedy)...")
            rag_prompts = [self.construct_rag_prompt(q, h) for q, h in zip(rag_questions, hints_list)]
            rag_outputs = self.llm.generate(rag_prompts, self.params_rag)
            
            # 2.4 填回结果
            for idx, rag_idx in enumerate(rag_indices):
                output = rag_outputs[idx]
                pred = self.extract_answer(output.outputs[0].text)
                
                final_results[rag_idx] = {
                    "id": rag_idx,
                    "question": rag_questions[idx],
                    "ground_truth": ground_truths[rag_idx],
                    "method": "Adaptive RAG (Recovered)",
                    "prediction": pred,
                    "raw_output": output.outputs[0].text,
                    "retrieved_trigger": hints_list[idx][0]['trigger'] if hints_list[idx] else None
                }

        # ==========================================
        # Stage 3: 统计最终分数
        # ==========================================
        print("\n📈 计算最终统计数据...")
        correct_count = 0
        rag_wins = 0
        rag_total = len(rag_indices)
        
        # 用来做对比：如果当时SC即使不一致也强行选众数会怎样？（作为 Baseline 对比）
        # 这里为了简单，我们只统计最终系统的准确率
        
        for res in final_results:
            gt = res['ground_truth']
            
            # 检查正确性
            # 注意：extract_answer 已经在前面做过了，prediction 是 float 或 None
            # check_correct 需要重新适配一下入参格式，或者直接在这里比对数字
            
            if "####" in gt: gold_val = self.extract_answer(gt.split("####")[1])
            else: gold_val = self.extract_answer(gt)
            
            pred_val = res['prediction']
            
            is_right = False
            if gold_val is not None and pred_val is not None:
                if abs(gold_val - pred_val) < 1e-4:
                    is_right = True
            
            if is_right:
                correct_count += 1
                if "RAG" in res['method']:
                    rag_wins += 1
            
            res['is_correct'] = is_right

        acc = correct_count / len(questions) * 100
        baseline_acc = baseline_correct / len(questions) * 100
        rag_recovery_rate = (rag_wins / len(questions) * 100) if rag_total > 0 else 0
        
        print("\n" + "="*50)
        print("🏆 自适应 RAG 评测报告")
        print("="*50)
        print(f"Total Questions      : {len(questions)}")
        print(f"Baseline Accuracy    : {baseline_acc:.2f}% (SC-3 Majority Vote)")
        print(f"Overall Accuracy     : {acc:.2f}%")
        print(f"RAG improved         : {rag_recovery_rate} %")
        print("-" * 50)
        print(f"RAG Activated        : {rag_total} cases")
        print(f"RAG Recovered (Win)  : {rag_wins} cases")
        print("="*50)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"📄 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass

    evaluator = AdaptiveEvaluator()
    evaluator.run_evaluation()