import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
import random
import time
import shutil
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# ================= 配置区域 =================
DEEPSEEK_API_KEY = "" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

STUDENT_MODEL_PATH = "Qwen/Qwen2.5-Math-0.5B"
DB_PATH = "./math_notebook_db"
SAMPLE_SIZE = 7000         # 增大样本量以发挥并发优势
MAX_RETRY_ROUNDS = 3       
TEACHER_CONCURRENCY = 15   # DeepSeek API 的并发线程数 (根据你的API Rate Limit调整)

# ================= 1. 知识库管理 =================
class MathNotebookDB:
    def __init__(self, reset=False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        # 最终的高质量经验库
        self.collection = self.client.get_or_create_collection(name="elite_strategies")
        
        # 临时错题检索库 (用于泛化验证)
        self.failed_collection = self.client.get_or_create_collection(name="temp_failed_cases")
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    def index_failed_cases(self, failed_cases):
        """将错题建立临时索引，用于后续的泛化验证"""
        if not failed_cases: return
        print("🧠 正在构建错题向量索引 (用于泛化验证)...")
        
        ids = [str(c['id']) for c in failed_cases]
        documents = [c['question'] for c in failed_cases]
        embeddings = self.embedder.encode(documents).tolist()
        # 存 GT 方便验证
        metadatas = [{"ground_truth": c['ground_truth']} for c in failed_cases]
        
        # 先清空旧的
        try:
            self.client.delete_collection("temp_failed_cases")
            self.failed_collection = self.client.create_collection("temp_failed_cases")
        except:
            pass
            
        self.failed_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search_similar_failed_case(self, trigger_text, exclude_id):
        """根据 Trigger 搜索相似的错题"""
        embedding = self.embedder.encode(trigger_text).tolist()
        # 搜 2 个，防止第 1 个是自己
        results = self.failed_collection.query(query_embeddings=[embedding], n_results=2)
        
        if not results['ids'][0]: return None
        
        for i, found_id in enumerate(results['ids'][0]):
            if str(found_id) != str(exclude_id):
                return {
                    "id": int(found_id),
                    "question": results['documents'][0][i],
                    "ground_truth": results['metadatas'][0][i]['ground_truth']
                }
        return None

    def save_experience_batch(self, experiences):
        if not experiences: return
        triggers = [e['trigger'] for e in experiences]
        embeddings = self.embedder.encode(triggers).tolist()
        ids = [f"exp_{int(time.time())}_{random.randint(10000,99999)}_{i}" for i in range(len(experiences))]
        documents = [e['rule_text'] for e in experiences]
        metadatas = [{"trigger": e['trigger'], "source_question": e['original_q'][:200]} for e in experiences]
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"💾 [入库] {len(experiences)} 条经验通过双重验证，已存入知识库")

# ================= 2. 导师代理 =================
class DeepSeekTeacher:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def _call_api_single(self, task_data):
        q = task_data['question']
        wrong_ans = task_data['wrong_ans']
        gt = task_data['ground_truth']
        prev_feedback = task_data.get('feedback')

        system_prompt = """You are a Meta-Cognitive Math Tutor specializing in "Knowledge Distillation". 
Your task is to analyze a specific failure by a Student Model and distill it into a **Universal Cognitive Schema** (Trigger + Strategy) that can be stored in a vector database to solve *any* similar future problems.

### GOAL
Transform a specific wrong answer into a high-level, abstract mathematical intuition.

### INPUT DATA
1. **Problem**: The specific math word problem.
2. **Student Wrong Answer**: The incorrect path taken.
3. **Correct Solution**: The ground truth logic.

### OUTPUT SECTIONS (Strict JSON Format)

#### 1. trigger_scenario (The "Search Key")
**Definition**: A concise, dense description of the *problem structure* and *key concepts* that would make an expert say, "Ah, this is a [Trigger] problem."
**Purpose**: This text will be embedded to retrieve this strategy later.
**Requirements**:
*   Focus on **structural patterns** (e.g., "Relative motion," "Compound ratios," "Work rate with delays").
*   Include key **entity relationships** (e.g., "Two objects moving towards each other," "Part-to-whole comparison").
*   **DO NOT** mention specific objects (like "apples", "cars") or numbers from the problem. Use general terms (entities, items, units).

#### 2. strategy_text (The "Algorithm")
**Definition**: A step-by-step, abstract algorithm to solve this class of problems.
**Requirements**:
*   **ABSTRACT**: Use variables ($N$, $X$, $T_{total}$) instead of specific numbers.
*   **IMPERATIVE**: Write as instructions (e.g., "1. Define variable X as... 2. Set up the equation...").
*   **LOGICAL**: Explain *how* to set up the relationships, not just the arithmetic.
*   **WARNING**: Explicitly point out the conceptual trap the student fell into (e.g., "Do not confuse individual time with total time").

### ONE-SHOT EXAMPLE
**Input Problem**: "John paints a fence in 3 hours. Tom paints it in 6 hours. How long if they work together?"
**Bad Trigger**: "Problem about John and Tom painting fences." (Too specific)
**Good Trigger**: "Work rate problem involving two agents working simultaneously with different individual rates."
**Bad Strategy**: "Divide 6 by 3 and add them." (Wrong and specific)
**Good Strategy**: "1. Determine individual rates: Rate_A = 1/Time_A and Rate_B = 1/Time_B. 2. Calculate combined rate: Rate_Total = Rate_A + Rate_B. 3. Solve for total time: Time_Total = 1 / Rate_Total. Warning: Do not average the times directly; always sum the rates."

### RESPONSE FORMAT
```json
{
    "trigger_scenario": "...",
    "strategy_text": "..."
}
"""
        user_content = f"[Problem]: {q}\n[Correct]: {gt}\n[Student Wrong]: {wrong_ans}"
        if prev_feedback:
            user_content += f"\n[Previous Failed Hint]: {prev_feedback}"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                temperature=0.7,
                response_format={"type": "json_object"},
                timeout=30
            )
            res_json = json.loads(response.choices[0].message.content)
            return {
                "id": task_data['id'],
                "strategy": res_json.get("strategy_text", ""),
                "trigger": res_json.get("trigger_scenario", ""),
                "success": True
            }
        except Exception as e:
            return {"id": task_data['id'], "success": False, "error": str(e)}

    def batch_teach(self, failed_cases_list):
        print(f"👨‍🏫 导师正在批量批改作业 (并发数: {TEACHER_CONCURRENCY})...")
        results = []
        with ThreadPoolExecutor(max_workers=TEACHER_CONCURRENCY) as executor:
            future_to_case = {executor.submit(self._call_api_single, case): case for case in failed_cases_list}
            for future in tqdm(as_completed(future_to_case), total=len(failed_cases_list), desc="DeepSeek Teaching"):
                results.append(future.result())
        return {r['id']: r for r in results if r['success']}

# ================= 3. 学生代理 =================
class QwenStudent:
    def __init__(self):
        print(f"🧑‍🎓 Qwen 学生初始化 (Batch Mode)...")
        self.llm = LLM(
            model=STUDENT_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            enforce_eager=True
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=512)

    def construct_prompt(self, question, hint=None):
        if hint:
            return f"""<|im_start|>user
Hint from Tutor: {hint}

Question: {question}
Please reason step-by-step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"<|im_start|>user\nQuestion: {question}\nPlease reason step-by-step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"

    def batch_solve(self, input_data):
        """
        input_data: List of dicts {'question': str, 'hint': str (optional)}
        """
        prompts = [self.construct_prompt(item['question'], item.get('hint')) for item in input_data]
        outputs = self.llm.generate(prompts, self.params, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

    def construct_abstraction_prompt(self, q):
        """
        [核心] 抽象化 Prompt
        目的是去除噪音，提取骨架，以便于和知识库中的 Trigger 匹配
        """
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

    def batch_abstraction(self, questions):
        """新增：批量抽象化题目"""
        prompts = []
        for q in questions:
            # 这里调用 construct_abstraction_prompt 方法
            # 注意：这个方法需要在外部定义或者作为类方法
            # 为了方便，这里直接内联 prompt 构造
            prompt = f"""<|im_start|>user
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
            prompts.append(prompt)
            
        outputs = self.llm.generate(prompts, self.params, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

# ================= 4. 主流程控制器 =================
class DistillationPipeline:
    def __init__(self):
        self.db = MathNotebookDB(reset=True)
        self.teacher = DeepSeekTeacher()
        self.student = QwenStudent()
        self.dataset = load_dataset("gsm8k", "main")['train']

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

    def run(self):
        # 1. 数据准备
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:SAMPLE_SIZE]
        
        batch_data = []
        for i in selected_indices:
            batch_data.append({
                "id": i,
                "question": self.dataset[i]['question'],
                "ground_truth": self.dataset[i]['answer'],
                "status": "pending", 
                "hint": None,
                "feedback": None
            })

        print(f"🚀 双重验证蒸馏开始 (N={SAMPLE_SIZE})...")

        # ================= Phase 1: 批量裸跑筛选错题 =================
        print("\n[Phase 1] 批量裸考 (Base Run)...")
        base_answers = self.student.batch_solve(batch_data)
        
        active_failed_cases = []
        for i, ans in enumerate(base_answers):
            item = batch_data[i]
            if not self.check_correct(ans, item['ground_truth']):
                item['wrong_ans'] = ans
                active_failed_cases.append(item)
        
        print(f"   -> 错题数: {len(active_failed_cases)} / {SAMPLE_SIZE}")

        # 将错题建立索引，供后续泛化验证使用
        # 🔴 [修改点] 建立索引时，先对错题进行抽象化，用抽象后的文本建索引
        print("🧠 正在对错题进行抽象化以构建索引...")
        failed_questions = [c['question'] for c in active_failed_cases]
        abstracted_questions = self.student.batch_abstraction(failed_questions)
        
        # 将抽象后的文本回填到 failed_cases 中，方便后续索引
        for i, c in enumerate(active_failed_cases):
            c['abstract_question'] = abstracted_questions[i]
            
        # 修改 MathNotebookDB.index_failed_cases 方法调用
        # 这里需要稍微修改 index_failed_cases 的实现逻辑，让它使用 abstract_question
        self.db.index_failed_cases(active_failed_cases)

        # ================= Phase 2: 循环蒸馏 =================
        for round_idx in range(1, MAX_RETRY_ROUNDS + 1):
            if not active_failed_cases: break
                
            print(f"\n[Round {round_idx}] 正在处理 {len(active_failed_cases)} 道错题...")

            # A. Teacher 介入
            teacher_results = self.teacher.batch_teach(active_failed_cases)
            
            ready_to_solve_cases = []
            for case in active_failed_cases:
                if case['id'] in teacher_results:
                    res = teacher_results[case['id']]
                    case['hint'] = res['strategy']
                    case['trigger'] = res['trigger'] 
                    ready_to_solve_cases.append(case)
            
            if not ready_to_solve_cases: break

            # B. Student 重做原题 (Primary Verification)
            print(f"   ✍️ 学生重做原题...")
            new_answers = self.student.batch_solve(ready_to_solve_cases)

            # C. 筛选原题做对的，准备进行泛化验证
            candidates_for_generalization = [] # (case, rule)
            still_failed_cases = []
            
            for i, ans in enumerate(new_answers):
                case = ready_to_solve_cases[i]
                if self.check_correct(ans, case['ground_truth']):
                    # 原题做对了，进入泛化候选队列
                    candidates_for_generalization.append(case)
                else:
                    # 还是错，更新 feedback
                    case['feedback'] = f"Hint: '{case['hint']}', Answer: '{ans}' (Wrong)."
                    case['wrong_ans'] = ans
                    still_failed_cases.append(case)

            # D. 泛化验证 (Generalization Verification)
            # 只有通过了这一步，才算真正的成功
            final_success_buffer = []
            
            if candidates_for_generalization:
                print(f"   ⚔️ 正在对 {len(candidates_for_generalization)} 条经验进行泛化验证...")
                
                # 1. 为每个 candidate 寻找相似错题 (Neighbor)
                verify_tasks = []
                for case in candidates_for_generalization:
                    # 🔴 [修改点] 使用 Trigger (已经是抽象的) 去搜 abstract_question 索引
                    neighbor = self.db.search_similar_failed_case(case['trigger'], exclude_id=case['id'])
                    if neighbor:
                        # 构造验证任务：用 case 的 hint 去解 neighbor 的 question
                        verify_tasks.append({
                            "question": neighbor['question'],
                            "hint": case['hint'], # 核心：使用原题的经验
                            "ground_truth": neighbor['ground_truth'],
                            "source_case": case # 关联回去以便保存
                        })
                    else:
                        pass

                # 2. 批量执行泛化验证
                if verify_tasks:
                    verify_answers = self.student.batch_solve(verify_tasks)
                    
                    for i, v_ans in enumerate(verify_answers):
                        task = verify_tasks[i]
                        if self.check_correct(v_ans, task['ground_truth']):
                            # 泛化验证通过！
                            final_success_buffer.append({
                                "original_q": task['source_case']['question'],
                                "rule_text": task['source_case']['hint'],
                                "trigger": task['source_case']['trigger']
                            })
            
            # E. 存入通过双重验证的经验
            if final_success_buffer:
                self.db.save_experience_batch(final_success_buffer)
            
            print(f"   -> 原题修复: {len(candidates_for_generalization)} | 泛化通过: {len(final_success_buffer)}")
            active_failed_cases = still_failed_cases

        print("\n" + "="*50)
        print("🏆 训练结束")

if __name__ == "__main__":
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass

    pipeline = DistillationPipeline()
    pipeline.run()