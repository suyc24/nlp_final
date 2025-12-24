import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import torch
import shutil
import sys
import numpy as np
import chromadb
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from sympy import simplify, parse_expr
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ThreadPoolExecutor
try:
    from latex2sympy2 import latex2sympy
except ImportError:
    latex2sympy = None

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct" 
DB_PATH = "./math_notebook_db"
OUTPUT_FILE = "math_adaptive_eval_result.json"
SAMPLE_SIZE = 5000      
SC_N = 3                
RAG_TOP_K = 1           
SIMILARITY_THRESHOLD = 0.70
EVALUATOR_CONCURRENCY = 32 

VALID_TYPES = [
    "Algebra", "Geometry", "Number Theory", "Counting & Probability", 
    "Precalculus", "Calculus", "Linear Algebra"
]

# ================= 1. 数学评测工具 (保持不变) =================
class MathEvaluator:
    def __init__(self, timeout=3):
        """
        :param timeout: SymPy 验证的超时时间(秒)
        """
        self.timeout = timeout

    def remove_boxed(self, s):
        if not s: return None
        if "\\boxed" not in s: return None
        idx = s.rfind("\\boxed{")
        if idx < 0: return None
        i = idx + len("\\boxed{")
        num_open = 1
        for j in range(i, len(s)):
            if s[j] == "{": num_open += 1
            elif s[j] == "}": num_open -= 1
            if num_open == 0: return s[idx + len("\\boxed{"):j]
        return None

    def _clean_latex(self, s):
        if not s: return ""
        s = str(s)
        replacements = [
            ("\\$", ""), ("\\text", ""), ("\\mbox", ""), ("\\mathrm", ""),
            ("\\,", ""), ("\\!", ""), ("\\ ", ""), ("%", ""),
            ("\\left", ""), ("\\right", ""), ("\\limits", ""), 
            ("°", "") 
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        s = s.replace("\\div", "/")
        s = s.replace("\\cdot", "*")
        s = "".join(s.split())
        return s

    def _sympy_check_core(self, pred_str, gt_str):
        pred_sym = None
        gt_sym = None
        if latex2sympy:
            try:
                pred_sym = latex2sympy(pred_str)
                gt_sym = latex2sympy(gt_str)
            except: pass

        if pred_sym is None or gt_sym is None:
            try:
                clean_pred = pred_str.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\frac", "")
                clean_gt = gt_str.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\frac", "")
                pred_sym = parse_expr(clean_pred)
                gt_sym = parse_expr(clean_gt)
            except: return False

        try:
            return simplify(pred_sym - gt_sym) == 0
        except: return False

    def verify(self, model_output, ground_truth):
        """
        主入口函数 (单条)
        """
        pred_inner = self.remove_boxed(model_output)
        gt_inner = self.remove_boxed(ground_truth)
        
        if gt_inner is None: gt_inner = ground_truth
        if pred_inner is None: return False, None

        norm_pred = self._clean_latex(pred_inner)
        norm_gt = self._clean_latex(gt_inner)

        # Level 1
        if norm_pred == norm_gt: return True, pred_inner
        # Level 2
        try:
            if abs(float(norm_pred) - float(norm_gt)) < 1e-4: return True, pred_inner
        except: pass
        # Level 3
        if "," in norm_pred and "," in norm_gt:
            try:
                set_pred = sorted([self._clean_latex(x) for x in pred_inner.split(',') if x.strip()])
                set_gt = sorted([self._clean_latex(x) for x in gt_inner.split(',') if x.strip()])
                if set_pred == set_gt: return True, pred_inner
            except: pass
        # Level 4
        try:
            is_equiv = func_timeout(self.timeout, self._sympy_check_core, args=(pred_inner, gt_inner))
            if is_equiv: return True, pred_inner
        except: pass

        return False, pred_inner

    def _verify_wrapper(self, args):
        """线程池辅助函数"""
        return self.verify(*args)

    def batch_verify(self, pairs):
        """
        [新增] 并行评测函数
        :param pairs: list of (model_output, ground_truth)
        :return: list of (is_correct, extracted_content)
        """
        if not pairs: return []
        print(f"⚖️ 正在并行评测 {len(pairs)} 条答案 (Threads={EVALUATOR_CONCURRENCY})...")
        
        results = []
        with ThreadPoolExecutor(max_workers=EVALUATOR_CONCURRENCY) as executor:
            # 使用 list(tqdm(...)) 来显示进度条
            results = list(tqdm(executor.map(self._verify_wrapper, pairs), total=len(pairs), desc="Evaluating"))
        return results

# ================= 2. 增强版检索器 =================
class KnowledgeRetriever:
    def __init__(self, db_path):
        print(f"📚 [Debug] 准备加载知识库路径: {db_path}")
        
        # 1. 初始化 ChromaDB
        print("   -> [1/3] 连接 ChromaDB Client...")
        try:
            # 强制设置 settings，防止一些 telemetry 导致的卡顿
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(
                path=db_path, 
                settings=Settings(anonymized_telemetry=False)
            )
            print("   -> [1/3] ChromaDB 连接成功")
        except Exception as e:
            print(f"❌ ChromaDB 连接失败: {e}")
            raise e

        # 2. 获取集合
        print("   -> [2/3] 获取集合 elite_strategies...")
        try:
            self.collection = self.client.get_collection(name="elite_strategies")
            count = self.collection.count()
            print(f"   -> [2/3] 集合获取成功，现有经验: {count} 条")
        except Exception as e:
            print(f"⚠️ 警告: 未找到 elite_strategies 集合，RAG 将失效。({e})")
            self.collection = None

        # 3. 加载 Embedding 模型
        print("   -> [3/3] 加载 Embedding 模型 (CPU模式)...")
        try:
            # 强制指定 device='cpu'，防止和 vLLM 抢 GPU 显存导致死锁
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
            print("   -> [3/3] Embedding 模型加载完毕")
        except Exception as e:
            print(f"❌ Embedding 模型加载失败: {e}")
            # 如果下载失败，可能是网络问题
            raise e
        
        print("✅ 检索器初始化完成！")

    def search_one(self, query_text, problem_type=None):
        # 保持原有的 search_one 逻辑不变
        if not query_text or self.collection is None: return []

        # 显式转为 numpy 只有在某些版本需要，这里保持原样即可
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True).tolist()
        
        # 策略 A: 如果有 Type，先尝试带 Filter 检索
        if problem_type:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=RAG_TOP_K,
                    where={"type": problem_type}
                )
                if results['ids'] and results['ids'][0]:
                    return self._process_results(results, 0)
            except Exception as e:
                pass

        # 策略 B: 全局检索 (Fallback)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=RAG_TOP_K
        )
        return self._process_results(results, 0)

    def _process_results(self, results, idx):
        valid = []
        if results['ids'][idx]:
            for j in range(len(results['ids'][idx])):
                dist = results['distances'][idx][j]
                if dist < SIMILARITY_THRESHOLD:
                    meta = results['metadatas'][idx][j]
                    valid.append({
                        "strategy": results['documents'][idx][j],
                        "trigger": meta.get('trigger', 'Unknown'),
                        "type": meta.get('type', 'Unknown'),
                        "dist": dist
                    })
        return valid

# ================= 3. 核心评测引擎 (多线程优化版) =================
class AdaptiveEvaluator:
    def __init__(self):
        print(f"🚀 初始化评测引擎: {MODEL_PATH}")
        self.evaluator = MathEvaluator()
        
        self.llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=False
        )
        
        self.params_sc = SamplingParams(n=SC_N, temperature=0.7, max_tokens=2048)
        self.params_greedy = SamplingParams(temperature=0.0, max_tokens=2048)
        self.params_abs = SamplingParams(temperature=0.0, max_tokens=300) 

    def construct_abstraction_prompt(self, q):
        return f"""<|im_start|>user
You are a Mathematics Librarian. Your task is to classify a math problem and abstract its core pattern for retrieval.

Step 1: **Classification**
Determine which ONE of the following categories best fits the problem:
{json.dumps(VALID_TYPES)}

Step 2: **Abstraction**
Identify the core mathematical structure (the "Trigger"). 
- Ignore specific numbers ($x=5$, 30 degrees).
- Use general terms (quadratic equation, inscribed circle, modular arithmetic).
- Describe *what* the problem is asking, not *how* to solve it.

[Example]
Problem: "Find the remainder when $7^{{2023}}$ is divided by 11."
Response:
Type: Number Theory
Pattern: Calculating the remainder of a large power modulo a prime number (Euler's/Fermat's Little Theorem).

[Target]
Problem: {q}
Response:<|im_end|>
<|im_start|>assistant
"""

    def parse_abstraction_output(self, text):
        p_type = None 
        p_pattern = text.strip()
        type_match = re.search(r"Type:\s*([a-zA-Z\s&]+)", text, re.IGNORECASE)
        if type_match:
            raw_type = type_match.group(1).strip()
            for vt in VALID_TYPES:
                if vt.lower() in raw_type.lower():
                    p_type = vt
                    break
        pat_match = re.search(r"Pattern:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if pat_match: p_pattern = pat_match.group(1).strip()
        p_pattern = p_pattern.replace("Type:", "").strip()
        return p_type, p_pattern

    def construct_base_prompt(self, q):
        return f"<|im_start|>user\nProblem: {q}\nPlease reason step by step and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"

    def construct_rag_prompt(self, q, hints):
        if not hints: return self.construct_base_prompt(q)
        strategies_str = ""
        for i, h in enumerate(hints):
            strategies_str += f"[Strategy {i+1}] (Type: {h['type']})\nTrigger: {h['trigger']}\nGuidance: {h['strategy']}\n\n"

        return f"""<|im_start|>user
Reference Knowledge:
{strategies_str}
Problem: {q}
Instruction:
1. Analyze if the "Reference Knowledge" is relevant to this problem's type and structure.
2. If relevant, apply the guidance logic.
3. Reason step by step and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

    def get_majority_vote(self, output_obj):
        answers = []
        raw_texts = []
        for o in output_obj.outputs:
            raw_texts.append(o.text)
            # 这里的 remove_boxed 只是简单提取，不做 SymPy 验证，所以很快，无需并行
            ans_content = self.evaluator.remove_boxed(o.text)
            if ans_content:
                norm_ans = self.evaluator._clean_latex(ans_content)
                answers.append(norm_ans)
            else:
                answers.append("no_answer")
        
        if not answers: return "no_answer", False, raw_texts[0]
        counts = Counter(answers)
        major_ans, count = counts.most_common(1)[0]
        best_idx = answers.index(major_ans)
        return major_ans, (count >= 2 and major_ans != "no_answer"), raw_texts[best_idx]

    def run_evaluation(self):
        retriever = KnowledgeRetriever(DB_PATH)
        print("📂 加载 MATH 数据集 (Test Split)...")
        # 注意：评测一定要用 test 集，严禁使用 train 集
        dataset = load_dataset("jeggers/competition_math", "original", split='test')
        
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)
        indices = indices[:SAMPLE_SIZE]
        
        batch_data = [dataset[i] for i in indices]
        questions = [d['problem'] for d in batch_data]
        ground_truths = [d['solution'] for d in batch_data]
        
        final_results = []
        
        # ================= Phase 1: SC-3 探测 (Batch Generation) =================
        print(f"\n⚡️ [Phase 1] Base Model SC-{SC_N} (N={len(questions)})...")
        prompts = [self.construct_base_prompt(q) for q in questions]
        outputs_sc = self.llm.generate(prompts, self.params_sc)
        
        # 1.1 收集投票结果 (CPU 轻量级)
        sc_results_buffer = [] # 暂存结果
        rag_queue = [] 
        rag_indices_map = {} # map rag_queue_idx -> original_idx
        
        print("   -> 处理投票结果...")
        for i, output in enumerate(outputs_sc):
            maj_norm_ans, is_consistent, best_raw_text = self.get_majority_vote(output)
            
            sc_results_buffer.append({
                "id": i,
                "best_raw_text": best_raw_text,
                "is_consistent": is_consistent,
                "gt": ground_truths[i]
            })
            
        # 1.2 并行评测 Baseline (CPU 密集型 -> 多线程)
        # 将 "生成" 和 "评测" 解耦，极大提高速度
        pairs_to_verify = [(item['best_raw_text'], item['gt']) for item in sc_results_buffer]
        verify_results_bools = self.evaluator.batch_verify(pairs_to_verify) # list of (bool, extracted)
        
        # 1.3 构建 Phase 1 结果
        for i, (is_correct, extracted) in enumerate(verify_results_bools):
            item = sc_results_buffer[i]
            orig_idx = item['id']
            
            record = {
                "id": orig_idx,
                "dataset_type": batch_data[orig_idx]['type'],
                "question": questions[orig_idx],
                "ground_truth": ground_truths[orig_idx],
                "baseline_pred": extracted,
                "baseline_correct": is_correct,
                "is_consistent": item['is_consistent'],
                "final_method": "Baseline (SC Consistent)",
                "final_pred": extracted,
                "final_correct": is_correct
            }
            
            if item['is_consistent']:
                final_results.append(record)
            else:
                record['final_method'] = "RAG (Inconsistent)"
                rag_queue.append(record)
        
        print(f"   -> 进入 RAG 流程: {len(rag_queue)}/{len(questions)}")

        baseline_acc = sum(1 for is_correct, _ in verify_results_bools if is_correct)

        # ================= Phase 2: Adaptive RAG (带 Type Filter) =================
        if rag_queue:
            rag_qs = [r['question'] for r in rag_queue]
            
            # 2.1 批量生成“分类+抽象” (GPU Batch)
            print(f"\n🌀 [Phase 2] Abstraction & Classification...")
            abs_prompts = [self.construct_abstraction_prompt(q) for q in rag_qs]
            abs_outputs = self.llm.generate(abs_prompts, self.params_abs)
            
            # 2.2 循环检索
            # 注意：检索本身是 I/O 密集型，也可以并行，但为了数据库安全这里维持串行或者用轻量并行
            # 考虑到 ChromaDB 的 Client 限制，这里保持 Loop 检索，但速度通常够快
            print(f"🔍 [Phase 2.1] Filtered Retrieval...")
            hints_list = []
            
            for idx, o in enumerate(tqdm(abs_outputs, desc="Retrieving")):
                llm_output = o.outputs[0].text
                p_type, p_pattern = self.parse_abstraction_output(llm_output)
                hints = retriever.search_one(p_pattern, problem_type=p_type)
                hints_list.append(hints)
                
                rag_queue[idx]['predicted_type'] = p_type
                rag_queue[idx]['abstraction'] = p_pattern

            # 2.3 RAG 推理 (GPU Batch)
            print(f"⚡️ [Phase 3] RAG Inference...")
            rag_prompts = [self.construct_rag_prompt(q, h) for q, h in zip(rag_qs, hints_list)]
            rag_outputs = self.llm.generate(rag_prompts, self.params_greedy)
            
            # 2.4 并行评测 RAG 结果 (CPU 多线程)
            rag_pairs_to_verify = []
            for j, output in enumerate(rag_outputs):
                rag_raw_text = output.outputs[0].text
                rag_pairs_to_verify.append((rag_raw_text, rag_queue[j]['ground_truth']))
                # 暂存 raw text 方便后面填回
                rag_queue[j]['rag_raw_text_temp'] = rag_raw_text 
            
            rag_verify_results = self.evaluator.batch_verify(rag_pairs_to_verify)
            
            # 2.5 填回结果
            for j, (is_correct_rag, extracted_rag) in enumerate(rag_verify_results):
                record = rag_queue[j]
                
                record['rag_pred'] = extracted_rag
                record['rag_correct'] = is_correct_rag
                record['retrieved_strategies'] = [h['strategy'] for h in hints_list[j]]
                record['final_pred'] = extracted_rag
                record['final_correct'] = is_correct_rag
                
                # 删除临时字段
                if 'rag_raw_text_temp' in record: del record['rag_raw_text_temp']
                
                final_results.append(record)

        # ================= Phase 3: 统计报告 =================
        final_results.sort(key=lambda x: x['id'])
        total = len(final_results)
        if total == 0:
            print("❌ 没有数据被评测")
            return

        final_acc = sum(1 for r in final_results if r['final_correct'])
        
        type_match_count = 0
        rag_count = len(rag_queue)
        for r in final_results:
            if 'predicted_type' in r and r['predicted_type']:
                if r['predicted_type'].lower() in r['dataset_type'].lower():
                    type_match_count += 1
        
        print("\n" + "="*60)
        print(f"🏆 MATH Benchmark Evaluation Report (N={total})")
        print("="*60)
        print(f"1. Baseline Accuracy         : {baseline_acc / total * 100:.2f}%")
        print(f"2. Adaptive RAG Accuracy     : {final_acc / total * 100:.2f}%")
        print(f"3. Net Improvement           : {(final_acc - baseline_acc) / total * 100:+.2f}%")
        print("-" * 60)
        if rag_count > 0:
            print(f"🔍 Type Classifier Accuracy  : {type_match_count / rag_count * 100:.2f}% (on {rag_count} cases)")
        print("="*60)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"📄 详细评测日志已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluator = AdaptiveEvaluator()
    evaluator.run_evaluation()