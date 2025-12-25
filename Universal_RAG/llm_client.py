
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from . import config
from . import prompts

class DeepSeekTeacher:
    def __init__(self):
        self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    def _call_api_single(self, task_data):
        q = task_data['question']
        wrong_ans = task_data['wrong_ans']
        gt = task_data['ground_truth']
        prev_feedback = task_data.get('feedback')

        user_content = f"[Problem]: {q}\n[Correct]: {gt}\n[Student Wrong]: {wrong_ans}"
        if prev_feedback:
            user_content += f"\n[Previous Failed Hint]: {prev_feedback}"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": prompts.TEACHER_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
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
        print(f"👨‍🏫 导师正在批量批改作业 (并发数: {config.TEACHER_CONCURRENCY})...")
        results = []
        with ThreadPoolExecutor(max_workers=config.TEACHER_CONCURRENCY) as executor:
            future_to_case = {executor.submit(self._call_api_single, case): case for case in failed_cases_list}
            for future in tqdm(as_completed(future_to_case), total=len(failed_cases_list), desc="DeepSeek Teaching"):
                results.append(future.result())
        return {r['id']: r for r in results if r['success']}

class StudentClient:
    def __init__(self, model_path=None):
        self.model_path = model_path or config.STUDENT_MODEL_PATH
        print(f"🚀 初始化评测引擎: {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True
        )
        # 默认 Greedy 参数
        self.params_greedy = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>", "<|endoftext|>"])
        # SC 参数
        self.params_sc = SamplingParams(n=config.SC_N, temperature=0.7, max_tokens=1024, stop=["<|im_end|>", "<|endoftext|>"])
        # 抽象化参数
        self.params_abs = SamplingParams(temperature=0.0, max_tokens=128)

    def batch_solve(self, input_data):
        """
        用于训练阶段的批量解题
        input_data: List of dicts {'question': str, 'hint': str (optional)}
        """
        prompts_list = []
        for item in input_data:
            if item.get('hint'):
                prompts_list.append(prompts.construct_hint_prompt(item['question'], item['hint']))
            else:
                prompts_list.append(prompts.construct_base_prompt(item['question']))
        
        outputs = self.llm.generate(prompts_list, self.params_greedy, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

    def batch_abstraction(self, questions):
        """批量抽象化题目"""
        prompts_list = [prompts.construct_abstraction_prompt(q) for q in questions]
        outputs = self.llm.generate(prompts_list, self.params_abs, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

    def generate_sc(self, questions):
        """Self-Consistency 生成"""
        prompts_list = [prompts.construct_base_prompt(q) for q in questions]
        return self.llm.generate(prompts_list, self.params_sc, use_tqdm=True)

    def generate_greedy(self, questions):
        """Greedy 生成 (用于 Baseline)"""
        prompts_list = [prompts.construct_base_prompt(q) for q in questions]
        return self.llm.generate(prompts_list, self.params_greedy, use_tqdm=True)

    def generate_rag(self, questions, hints_list):
        """RAG 生成"""
        prompts_list = [prompts.construct_rag_prompt(q, h) for q, h in zip(questions, hints_list)]
        return self.llm.generate(prompts_list, self.params_greedy, use_tqdm=True)
