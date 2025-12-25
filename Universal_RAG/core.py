
import random
from collections import Counter
from tqdm import tqdm
from . import config
from .db_manager import MathNotebookDB
from .llm_client import DeepSeekTeacher, StudentClient
from .utils import default_check_correct, extract_answer

class PrincipleRAGModel:
    def __init__(self, db_path=None, model_path=None):
        self.db = MathNotebookDB(db_path=db_path)
        self.student = StudentClient(model_path=model_path)
        # Teacher 仅在训练时按需初始化，或者这里直接初始化
        self.teacher = DeepSeekTeacher()

    def train(self, training_data, verifier_func=None):
        """
        执行双重验证蒸馏训练流程。
        """
        # 0. 初始化验证器
        if verifier_func is None:
            verifier_func = default_check_correct

        # 1. 数据预处理
        batch_data = []
        for i, item in enumerate(training_data):
            data_item = item.copy()
            if 'id' not in data_item:
                data_item['id'] = i
            data_item.update({
                "status": "pending",
                "hint": None,
                "feedback": None
            })
            batch_data.append(data_item)

        print(f"🚀 双重验证蒸馏开始 (N={len(batch_data)})...")

        # ================= Phase 1: 批量裸跑筛选错题 =================
        print("\n[Phase 1] 批量裸考 (Base Run)...")
        base_answers = self.student.batch_solve(batch_data)
        
        active_failed_cases = []
        for i, ans in enumerate(base_answers):
            item = batch_data[i]
            if not verifier_func(ans, item['ground_truth']):
                item['wrong_ans'] = ans
                active_failed_cases.append(item)
        
        print(f"   -> 错题数: {len(active_failed_cases)} / {len(batch_data)}")

        # 将错题建立索引，供后续泛化验证使用
        print("🧠 正在对错题进行抽象化以构建索引...")
        failed_questions = [c['question'] for c in active_failed_cases]
        abstracted_questions = self.student.batch_abstraction(failed_questions)
        
        for i, c in enumerate(active_failed_cases):
            c['abstract_question'] = abstracted_questions[i]
            
        self.db.index_failed_cases(active_failed_cases)

        # ================= Phase 2: 循环蒸馏 =================
        for round_idx in range(1, config.MAX_RETRY_ROUNDS + 1):
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
            candidates_for_generalization = [] 
            still_failed_cases = []
            
            for i, ans in enumerate(new_answers):
                case = ready_to_solve_cases[i]
                if verifier_func(ans, case['ground_truth']):
                    candidates_for_generalization.append(case)
                else:
                    case['feedback'] = f"Hint: '{case['hint']}', Answer: '{ans}' (Wrong)."
                    case['wrong_ans'] = ans
                    still_failed_cases.append(case)

            # D. 泛化验证 (Generalization Verification)
            final_success_buffer = []
            
            if candidates_for_generalization:
                print(f"   ⚔️ 正在对 {len(candidates_for_generalization)} 条经验进行泛化验证...")
                
                # 1. 为每个 candidate 寻找相似错题 (Neighbor)
                verify_tasks = []
                for case in candidates_for_generalization:
                    neighbor = self.db.search_similar_failed_case(case['trigger'], exclude_id=case['id'])
                    if neighbor:
                        verify_tasks.append({
                            "question": neighbor['question'],
                            "hint": case['hint'], 
                            "ground_truth": neighbor['ground_truth'],
                            "source_case": case 
                        })

                # 2. 批量执行泛化验证
                if verify_tasks:
                    verify_answers = self.student.batch_solve(verify_tasks)
                    
                    for i, v_ans in enumerate(verify_answers):
                        task = verify_tasks[i]
                        if verifier_func(v_ans, task['ground_truth']):
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

    def predict(self, questions, force_rag=False, baseline_require=False):
        """
        执行 Adaptive RAG 预测流程。
        - Baseline 1 = SC Majority Vote (即使不一致也取众数)
        - Baseline 2 = Greedy
        - Final = SC Consistent 直接采纳，SC Inconsistent 进入 RAG
        """
        results = [None] * len(questions)
        
        # ================= Phase 0: Baseline 2 (Direct Greedy) =================
        greedy_outputs = None
        if baseline_require:
            print(f"\n📉 [Phase 0] 运行 Direct Greedy Baseline (N={len(questions)})...")
            greedy_outputs = self.student.generate_greedy(questions)

        # ================= Phase 1: SC-3 投票 =================
        print(f"\n⚡️ [Phase 1] 运行 SC-{config.SC_N} 投票 (N={len(questions)})...")
        outputs_sc = self.student.generate_sc(questions)
        
        rag_indices = []
        rag_questions = []
        
        consistent_count = 0
        
        for i, output in enumerate(outputs_sc):
            maj_ans, is_consistent, raw_text = self._get_majority_vote(output)
            
            # 初始化结果对象
            res = {
                "question": questions[i],
                "retrieved_context": []
            }
            
            # 填充 Baseline 信息
            if baseline_require:
                # Baseline 2: Greedy
                if greedy_outputs:
                    greedy_raw = greedy_outputs[i].outputs[0].text
                    greedy_pred = extract_answer(greedy_raw)
                    res['baseline_2_greedy'] = greedy_pred
                    res['baseline_2_raw'] = greedy_raw
                
                # Baseline 1: SC Majority Vote (即使不一致也取众数)
                res['baseline_1_majority'] = maj_ans
                res['baseline_1_raw'] = raw_text

            if is_consistent and not force_rag:
                # SC一致，直接采纳
                consistent_count += 1
                res['prediction'] = maj_ans
                res['raw_output'] = raw_text
                res['method'] = "SC-3 (Consistent)"
                results[i] = res
            else:
                # SC不一致，进入RAG队列
                rag_indices.append(i)
                rag_questions.append(questions[i])
                results[i] = res # 先占位，后续更新 prediction
        
        print(f"   -> 一致性通过: {consistent_count}/{len(questions)}")
        print(f"   -> 需要 RAG 介入: {len(rag_questions)}/{len(questions)}")

        # ================= Phase 2: Adaptive RAG =================
        if rag_questions:
            print(f"\n🌀 [Phase 2.1] 对 {len(rag_questions)} 道难题进行抽象化...")
            abstract_queries = self.student.batch_abstraction(rag_questions)
            
            print(f"\n🔍 [Phase 2.2] 检索经验...")
            hints_list = self.db.batch_search(abstract_queries, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD)
            
            print(f"\n⚡️ [Phase 2.3] 运行 RAG 推理 (Greedy)...")
            rag_outputs = self.student.generate_rag(rag_questions, hints_list)
            
            for idx, rag_idx in enumerate(rag_indices):
                output = rag_outputs[idx]
                raw_text = output.outputs[0].text
                pred = extract_answer(raw_text) 
                
                # 更新 RAG 结果
                results[rag_idx]['prediction'] = pred
                results[rag_idx]['raw_output'] = raw_text
                results[rag_idx]['method'] = "Adaptive RAG (Recovered)"
                results[rag_idx]['retrieved_context'] = hints_list[idx]
                results[rag_idx]['retrieved_trigger'] = hints_list[idx][0]['trigger'] if hints_list[idx] else None
                
        return results

    def _get_majority_vote(self, output_obj):
        """
        辅助函数：处理 SC 投票
        """
        answers = []
        raw_texts = []
        for o in output_obj.outputs:
            raw_texts.append(o.text)
            val = extract_answer(o.text)
            if val is not None:
                answers.append(val)
        
        if not answers: return None, False, raw_texts[0]
        
        counts = Counter(answers)
        major_ans, count = counts.most_common(1)[0]
        
        # SC=3 时，>=2 算一致
        is_consistent = (count >= 2)
        
        # 找到对应 major_ans 的原始文本
        best_raw_text = raw_texts[0]
        for i, val in enumerate(answers):
            if val == major_ans:
                best_raw_text = raw_texts[i]
                break
                
        return major_ans, is_consistent, best_raw_text
