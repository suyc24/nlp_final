import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import math
from sympy import simplify, parse_expr
from sympy.parsing.latex import parse_latex # 基础版
try:
    from latex2sympy2 import latex2sympy # 增强版，建议安装
except ImportError:
    latex2sympy = None

from func_timeout import func_timeout, FunctionTimedOut
import signal
import json
import httpx
import time
import random
import shutil
import torch
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import chromadb
from datasets import load_dataset

# ================= 配置区 =================
# 你指定的配置
STUDENT_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
DB_PATH = "./math_notebook_db"
FAILED_CACHE_FILE = "failed_cases_checkpoint.json"
MAX_RETRY_ROUNDS = 3       
TEACHER_CONCURRENCY = 15  
EVALUATOR_CONCURRENCY = 32
SAMPLE_SIZE = 7500

DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

VOLC_API_KEY =""
VOLC_ENDPOINT_ID = ""

VALID_TYPES = [
    "Algebra", "Geometry", "Number Theory", "Counting & Probability", 
    "Precalculus", "Calculus", "Linear Algebra"
]

# ================= 1. 并行化数学评测器 (CPU 密集型) =================
class MathEvaluator:
    def __init__(self, timeout=3):
        """
        :param timeout: 单个题目 SymPy 验证的超时时间(秒)
        """
        self.timeout = timeout

    def remove_boxed(self, s):
        """提取 \boxed{...} 内容"""
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
        """标准化 LaTeX 字符串"""
        if not s: return ""
        s = str(s)
        replacements = [
            ("\\$", ""), ("\\text", ""), ("\\mathrm", ""), ("\\ ", ""), ("%", ""),
            ("\\left", ""), ("\\right", ""), ("\\limits", ""), ("°", "")
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\div", "/")
        return "".join(s.split())

    def _is_number(self, s):
        try:
            float(s)
            return True
        except:
            return False

    def _sympy_check_logic(self, pred_str, gt_str):
        """实际执行 SymPy 验证的逻辑"""
        pred_sym, gt_sym = None, None
        
        # 1. 尝试使用 latex2sympy2
        if latex2sympy:
            try:
                pred_sym = latex2sympy(pred_str)
                gt_sym = latex2sympy(gt_str)
            except:
                pass
        
        # 2. 回退到 SymPy parse_expr
        if pred_sym is None:
            try:
                # 简单的字符替换以适配 SymPy 语法
                clean_pred = pred_str.replace("^", "**").replace("{", "(").replace("}", ")")
                clean_gt = gt_str.replace("^", "**").replace("{", "(").replace("}", ")")
                pred_sym = parse_expr(clean_pred)
                gt_sym = parse_expr(clean_gt)
            except:
                return False
            
        if pred_sym is None or gt_sym is None:
            return False

        # 3. 核心判断
        return simplify(pred_sym - gt_sym) == 0

    def verify_single(self, task_tuple):
        """
        单个题目验证函数，用于线程池调用
        task_tuple: (prediction, ground_truth)
        """
        pred, gt = task_tuple
        
        pred_inner = self.remove_boxed(pred)
        gt_inner = self.remove_boxed(gt)
        if gt_inner is None: gt_inner = gt # Handle pure text GT
        if pred_inner is None: return False

        norm_pred = self._clean_latex(pred_inner)
        norm_gt = self._clean_latex(gt_inner)

        # Level 1: String Match
        if norm_pred == norm_gt: return True

        # Level 2: Set Match (e.g. "1, 2" == "2, 1")
        if "," in norm_pred and "," in norm_gt:
            try:
                if sorted(norm_pred.split(',')) == sorted(norm_gt.split(',')): return True
            except: pass

        # Level 3: Numeric Match
        if self._is_number(norm_pred) and self._is_number(norm_gt):
            if abs(float(norm_pred) - float(norm_gt)) < 1e-4: return True

        # Level 4: SymPy Match (with timeout)
        try:
            return func_timeout(self.timeout, self._sympy_check_logic, args=(pred_inner, gt_inner))
        except:
            return False

    def batch_verify(self, pred_gt_pairs):
        """
        🚀 并行评测入口
        """
        if not pred_gt_pairs: return []
        print(f"⚖️ 正在并行评测 {len(pred_gt_pairs)} 道题目 (CPU Threads: {EVALUATOR_CONCURRENCY})...")
        results = [False] * len(pred_gt_pairs)
        
        with ThreadPoolExecutor(max_workers=EVALUATOR_CONCURRENCY) as executor:
            futures = [executor.submit(self.verify_single, pair) for pair in pred_gt_pairs]
            # 使用 tqdm 监控评测进度
            for i, future in enumerate(tqdm(futures, desc="Evaluating")):
                results[i] = future.result()
        return results

# ================= 2. 知识库管理 =================
class MathNotebookDB:
    def __init__(self, reset=False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="elite_strategies")
        self.failed_collection = self.client.get_or_create_collection(name="temp_failed_cases")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    def index_failed_cases(self, failed_cases):
        if not failed_cases: return
        print("🧠 正在构建错题向量索引...")
        
        # 必须转为 str
        ids = [str(c['id']) for c in failed_cases]
        documents = []
        metadatas = []
        
        for c in failed_cases:
            # 优先使用抽象后的 Pattern 建索引
            doc_text = c.get('abstraction_pattern', c['question']) 
            documents.append(doc_text)
            
            meta = {
                "ground_truth": str(c['ground_truth']),
                "type": c.get('abstraction_type', 'Unknown'), # 存入类型，方便后续按类型过滤
                "original_question": c['question'] # 把原题存进 metadata，方便取回
            }
            metadatas.append(meta)
        
        embeddings = self.embedder.encode(documents).tolist()
        
        try:
            self.client.delete_collection("temp_failed_cases")
            self.failed_collection = self.client.create_collection("temp_failed_cases")
        except:
            pass
            
        self.failed_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    # [修改] 搜索逻辑
    def search_similar_failed_case(self, trigger_text, exclude_id, filter_type=None):
        """
        :param filter_type: (可选) 强制只在同类型题目中搜索
        """
        embedding = self.embedder.encode(trigger_text).tolist()
        
        # 构造过滤条件
        where_clause = None
        if filter_type:
            where_clause = {"type": filter_type}

        results = self.failed_collection.query(
            query_embeddings=[embedding], 
            n_results=5,
            where=where_clause # 支持类型过滤
        )
        
        if not results['ids'] or not results['ids'][0]: return None
        
        for i, found_id in enumerate(results['ids'][0]):
            if str(found_id) != str(exclude_id):
                return {
                    "id": found_id,
                    # 注意：documents里现在是 Pattern，原题在 metadata 里
                    "question": results['metadatas'][0][i]['original_question'], 
                    "ground_truth": results['metadatas'][0][i]['ground_truth'],
                    "type": results['metadatas'][0][i]['type']
                }
        return None

    def save_experience_batch(self, experiences):
        if not experiences: return
        ids = [f"exp_{int(time.time())}_{random.randint(10000,99999)}_{i}" for i in range(len(experiences))]
        triggers = [e['trigger'] for e in experiences]
        documents = [e['rule_text'] for e in experiences]
        # 这里把 Type 也存进去了
        metadatas = [{
            "trigger": e['trigger'], 
            "source_question": e['original_q'][:200],
            "type": e['type']
        } for e in experiences]
        
        self.collection.add(ids=ids, embeddings=self.embedder.encode(triggers).tolist(), documents=documents, metadatas=metadatas)
        print(f"💾 [入库] {len(experiences)} 条经验已存入知识库")

# ================= 3. 导师代理 (DeepSeek) =================
class DeepSeekTeacher:
    def __init__(self):
        http_client = httpx.Client(timeout=60.0) 
        self.client = OpenAI(
            api_key=VOLC_API_KEY,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=120.0,
            http_client=http_client
        )

    def _call_api_single(self, task_data):
        q = task_data['question']
        wrong_ans = task_data['wrong_ans']
        gt = task_data['ground_truth']
        prev_feedback = task_data.get('feedback')

        # 完整的 System Prompt，包含 Type 分类指令
        system_prompt = """You are a Fields Medal-level Mathematician acting as a "Cognitive Schema Distiller".
Your goal is to diagnose why a student model failed a complex math problem and distill a **Universal Abstract Schema** that can solve this class of problems.

### INPUT DATA
1. **Problem**: A competition-level math problem (LaTeX format).
2. **Student Wrong Answer**: The incorrect derivation or result.
3. **Correct Solution**: The ground truth proof/solution.

### OUTPUT REQUIREMENTS (Strict JSON)

#### 1. type
**Classify the problem into ONE of these categories**:
["Algebra", "Geometry", "Number Theory", "Counting & Probability", "Precalculus", "Calculus", "Linear Algebra"]

#### 2. trigger_scenario
A concise, structural description of the problem pattern.
*   **BAD**: "A problem about triangle ABC with side 3."
*   **GOOD**: "Geometry: Calculating area of a triangle given two sides and the included angle (SAS)."
*   **NOTE**: Do not use specific numbers. Use mathematical terms.

#### 3. strategy_text
An abstract, algorithmic guide.
*   **Must be step-by-step.**
*   **Must use variables** ($a, b, n$) instead of numbers.
*   **Must highlight the trap** that caused the error.
*   **Format**: "1. Identify variables... 2. Apply Theorem X... Warning: Check discriminant condition."

### RESPONSE FORMAT
```json
{
    "type": "Algebra",
    "trigger_scenario": "...",
    "strategy_text": "..."
}
```
"""
        user_content = f"[Problem]: {q}\n[Correct Solution]: {gt}\n[Student Wrong Trace]: {wrong_ans}"
        if prev_feedback:
            user_content += f"\n[Previous Failed Strategy]: {prev_feedback} (This hint did not work, please refine)"

        try:
            response = self.client.chat.completions.create(
                model=VOLC_ENDPOINT_ID,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                temperature=0.5, 
                response_format={"type": "json_object"},
                timeout=120.0
            )
            res_json = json.loads(response.choices[0].message.content)
            return {
                "id": task_data['id'],
                "type": res_json.get("type", "Algebra"), # 获取分类结果
                "strategy": res_json.get("strategy_text", ""),
                "trigger": res_json.get("trigger_scenario", ""),
                "success": True
            }
        except Exception as e:
            return {"id": task_data['id'], "success": False, "error": str(e)}

    def batch_teach(self, failed_cases_list):
        print(f"👨‍🏫 导师正在批量诊断 (并发数: {TEACHER_CONCURRENCY})...")
        results = {}
        with ThreadPoolExecutor(max_workers=TEACHER_CONCURRENCY) as executor:
            future_to_case = {executor.submit(self._call_api_single, case): case for case in failed_cases_list}
            for future in tqdm(as_completed(future_to_case), total=len(failed_cases_list), desc="DeepSeek Teaching"):
                r = future.result()
                if r['success']:
                    results[r['id']] = r
        return results

# ================= 4. 学生代理 (Qwen-Math) =================
class QwenStudent:
    def __init__(self):
        # 自动获取当前机器的 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"🧑‍🎓 Qwen-Math 学生初始化 (GPUs={gpu_count}, vLLM Tensor Parallel)...")
        
        self.llm = LLM(
            model=STUDENT_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=gpu_count, # 自动利用多卡
            gpu_memory_utilization=0.92, 
            max_model_len=8192, 
            enforce_eager=False
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=2048)

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

    # [新增] 批量抽象化函数
    def batch_abstract(self, questions):
        """
        输入: 题目列表
        输出: list of dict {'type': ..., 'pattern': ...}
        """
        prompts = [self.construct_abstraction_prompt(q) for q in questions]
        # 这里 max_tokens 不用太长，抽象描述通常很短
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=True)
        
        results = []
        for output in outputs:
            text = output.outputs[0].text
            # 简单的解析逻辑
            p_type = "Algebra"
            p_pattern = text
            
            # 解析 Type
            type_match = re.search(r"Type:\s*([a-zA-Z\s&]+)", text, re.IGNORECASE)
            if type_match: p_type = type_match.group(1).strip()
            
            # 解析 Pattern
            pat_match = re.search(r"Pattern:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
            if pat_match: p_pattern = pat_match.group(1).strip()
            
            results.append({"type": p_type, "pattern": p_pattern})
        return results

    def construct_prompt(self, question, hint=None):
        content = ""
        if hint:
            content += f"Hint: {hint}\n\n"
        content += f"Problem: {question}\n\nPlease reason step by step and put your final answer within \\boxed{{}}."
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def batch_solve(self, input_data):
        """
        利用 vLLM 的 Continuous Batching 进行极速推理。
        input_data: list of dict {'question': ..., 'hint': ...}
        """
        prompts = [self.construct_prompt(item['question'], item.get('hint')) for item in input_data]
        outputs = self.llm.generate(prompts, self.params, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

# ================= 5. 主流程控制器 =================
class DistillationPipeline:
    def __init__(self):
        self.db = MathNotebookDB(reset=False)
        self.evaluator = MathEvaluator() # 包含并行评测功能
        self.teacher = DeepSeekTeacher()
        self.student = QwenStudent()
        
        print("📚 Loading MATH Dataset (jeggers/competition_math)...")
        self.dataset = load_dataset("jeggers/competition_math", "original", split='train')

    def run(self):
        # 定义缓存文件名
        FAILED_CACHE_FILE = "failed_cases_checkpoint.json"
        
        active_failed_cases = []
        loaded_from_cache = False

        # ================= 1. 尝试读取本地缓存 (断点续传) =================
        if os.path.exists(FAILED_CACHE_FILE):
            print(f"\n📂 检测到本地错题缓存: {FAILED_CACHE_FILE}")
            print("⏩ 正在加载缓存，跳过 [Phase 1] 学生裸考阶段...")
            try:
                with open(FAILED_CACHE_FILE, "r", encoding="utf-8") as f:
                    active_failed_cases = json.load(f)
                print(f"✅ 成功加载 {len(active_failed_cases)} 道错题。")
                loaded_from_cache = True
            except Exception as e:
                print(f"❌ 缓存加载失败 ({e})，将重新运行 Phase 1。")

        # ================= 2. 如果没缓存，正常运行 Phase 1 (挖掘错题) =================
        if not loaded_from_cache:
            total_len = len(self.dataset)
            indices = list(range(total_len))
            random.shuffle(indices) 
            
            # 处理 SAMPLE_SIZE
            if SAMPLE_SIZE:
                indices = indices[:SAMPLE_SIZE]
                print(f"⚠️ Debug Mode: 仅采样 {SAMPLE_SIZE} 条数据")
        
            batch_data = []
            print(f"📦 正在加载数据...")
            
            for i in indices:
                row = self.dataset[i]
                batch_data.append({
                    "id": f"math_{i}",
                    "question": row['problem'],
                    "ground_truth": row['solution'],
                    "original_type": row.get('type', 'Unknown'), 
                    "level": row.get('level', 'Unknown'),
                    "status": "pending", 
                    "hint": None,
                    "feedback": None
                })
            
            print(f"🚀 高难度数学蒸馏开始 (Full Mode, N={len(batch_data)})...")

            # --- Phase 1: 批量裸考 ---
            print("\n[Phase 1] 批量裸考 (Base Run)...")
            # GPU 并行
            base_answers = self.student.batch_solve(batch_data)
            
            # CPU 并行评测
            verify_pairs = [(ans, item['ground_truth']) for ans, item in zip(base_answers, batch_data)]
            verify_results = self.evaluator.batch_verify(verify_pairs)
            
            for i, is_correct in enumerate(verify_results):
                item = batch_data[i]
                if not is_correct:
                    item['wrong_ans'] = base_answers[i]
                    active_failed_cases.append(item)
            
            print(f"   -> 初始准确率: {1 - (len(active_failed_cases) / len(batch_data)):.2%} (错题数: {len(active_failed_cases)})")

            # 初次保存缓存 (防止后续步骤崩溃)
            try:
                with open(FAILED_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(active_failed_cases, f, ensure_ascii=False, indent=2)
            except: pass

        # ================= [新增] 3. 学生自我抽象 (Self-Abstraction) =================
        # 在建立索引前，检查错题是否已经包含了抽象Pattern。如果没有，让学生生成。
        # 这一步是为了让检索库里的 key 变成抽象的数学 Pattern，而不是具体的数字题目。
        
        cases_to_abstract = [c for c in active_failed_cases if 'abstraction_pattern' not in c]
        
        if cases_to_abstract:
            print(f"\n🌀 正在对 {len(cases_to_abstract)} 道错题进行抽象化预处理...")
            # 提取题目文本
            raw_questions = [c['question'] for c in cases_to_abstract]
            
            # 调用 Student 的抽象能力 (需确保 QwenStudent 类中有 batch_abstract 方法)
            abs_results = self.student.batch_abstract(raw_questions)
            
            # 回填结果
            for c, res in zip(cases_to_abstract, abs_results):
                c['abstraction_type'] = res['type']     # 学生认为的类型
                c['abstraction_pattern'] = res['pattern'] # 学生提取的Pattern
            
            # 更新缓存文件 (保存宝贵的抽象结果)
            print(f"💾 更新错题缓存 (含抽象信息)...")
            with open(FAILED_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(active_failed_cases, f, ensure_ascii=False, indent=2)
        else:
            print("✅ 所有错题已包含抽象信息，跳过抽象化步骤。")

        # ================= 4. 建立错题索引 (基于抽象 Pattern) =================
        # 这里会调用修改后的 index_failed_cases，索引 abstraction_pattern
        self.db.index_failed_cases(active_failed_cases)

        # ================= Phase 2: 循环蒸馏 (保持不变) =================
        for round_idx in range(1, MAX_RETRY_ROUNDS + 1):
            if not active_failed_cases: break
                
            print(f"\n[Round {round_idx}] 正在攻克 {len(active_failed_cases)} 道难题...")

            # A. Teacher 介入 (生成高质量 Trigger 和 Strategy)
            teacher_results = self.teacher.batch_teach(active_failed_cases)
            
            ready_to_solve_cases = []
            for case in active_failed_cases:
                if case['id'] in teacher_results:
                    res = teacher_results[case['id']]
                    # 这里保存了 Teacher 生成的 type, trigger, strategy
                    case['hint'] = res['strategy']
                    case['trigger'] = res['trigger'] 
                    case['type'] = res['type'] 
                    ready_to_solve_cases.append(case)
            
            if not ready_to_solve_cases: break

            # B. Student 重做原题 (Verification A)
            print(f"   ✍️ Student 尝试应用新策略解原题...")
            new_answers = self.student.batch_solve(ready_to_solve_cases)
            
            # 批量验证重做结果
            verify_pairs = [(ans, c['ground_truth']) for ans, c in zip(new_answers, ready_to_solve_cases)]
            verify_results = self.evaluator.batch_verify(verify_pairs)

            candidates_for_generalization = [] 
            still_failed_cases = []
            
            for i, is_correct in enumerate(verify_results):
                case = ready_to_solve_cases[i]
                ans = new_answers[i]
                
                if is_correct:
                    candidates_for_generalization.append(case)
                else:
                    case['feedback'] = f"Strategy failed. Student output: {ans[-200:]}..." 
                    case['wrong_ans'] = ans
                    still_failed_cases.append(case)
            
            print(f"   -> 原题修复率: {len(candidates_for_generalization)}/{len(ready_to_solve_cases)}")

            # C. 泛化验证 (Verification B)
            final_success_buffer = []
            
            if candidates_for_generalization:
                print(f"   ⚔️ 进入泛化验证门控 (Based on Abstract Pattern)...")
                
                verify_tasks = []
                for case in candidates_for_generalization:
                    # [修改] 使用 filter_type 进行更精准的检索
                    # 逻辑：用 Teacher 生成的 Trigger 去搜 Student 抽象出的 Pattern库
                    # 并且强制要求题目类型一致 (case['type'] 来自 Teacher)
                    neighbor = self.db.search_similar_failed_case(
                        trigger_text=case['trigger'], 
                        exclude_id=case['id'],
                        filter_type=case.get('type') 
                    )
                    
                    # [回退机制] 如果同类型没搜到，尝试不限制类型搜一次
                    if not neighbor:
                        neighbor = self.db.search_similar_failed_case(
                            trigger_text=case['trigger'], 
                            exclude_id=case['id'],
                            filter_type=None
                        )

                    if neighbor:
                        verify_tasks.append({
                            "question": neighbor['question'], # 这里取出来的是原题文本
                            "hint": case['hint'],             # 使用原题的 Strategy
                            "ground_truth": neighbor['ground_truth'],
                            "source_case": case 
                        })

                if verify_tasks:
                    # GPU 并行解泛化题
                    verify_answers = self.student.batch_solve(verify_tasks)
                    
                    # CPU 并行验证泛化结果
                    v_pairs = [(ans, t['ground_truth']) for ans, t in zip(verify_answers, verify_tasks)]
                    v_results = self.evaluator.batch_verify(v_pairs)
                    
                    for i, is_correct in enumerate(v_results):
                        task = verify_tasks[i]
                        if is_correct:
                            final_success_buffer.append({
                                "original_q": task['source_case']['question'],
                                "rule_text": task['source_case']['hint'],
                                "trigger": task['source_case']['trigger'],
                                "type": task['source_case']['type'] # 存入 Teacher 确定的类型
                            })
            
            # D. 入库
            if final_success_buffer:
                self.db.save_experience_batch(final_success_buffer)
            
            active_failed_cases = still_failed_cases

        print("\n" + "="*50)
        print(f"🏆 训练结束。数据库中现有 {self.db.collection.count()} 条高价值数学策略。")

if __name__ == "__main__":
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass

    pipeline = DistillationPipeline()
    pipeline.run()

    print("👋 所有任务完成，正在强制关闭 vLLM 进程...")
    sys.exit(0)
