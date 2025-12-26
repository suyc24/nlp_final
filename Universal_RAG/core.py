
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
        """
        results = [{"question": question} for question in questions]

        # [Phase 0] Baseline 1: Greedy
        greedy_outputs = None
        if baseline_require:
            print(f"\n📉 [Phase 0] 运行 Direct Greedy Baseline (N={len(questions)})...")
            greedy_outputs = self.student.generate_greedy(questions)

        # [Phase 1] SC-N Majority Vote
        print(f"\n⚡️ [Phase 1] 运行 SC-{config.SC_N} 投票 (N={len(questions)})...")
        outputs_sc = self.student.generate_sc(questions)
        
        # ⚠️ 修改：分别存储元组（用于索引）和纯文本（用于LLM）
        rag_data_tuples = [] # [(question_text, original_index), ...]
        rag_indices = []     # [original_index, ...]

        for i, output in enumerate(outputs_sc):
            # 获取 SC 结果
            sc_outputs, is_consistent, sc_raw_text = self._get_majority_vote(output)
            results[i]["is_consistent"] = is_consistent
            
            # 默认 Final Answer 为 SC 结果
            results[i]["final_answer"] = sc_outputs
            results[i]["final_answer_raw"] = sc_raw_text
            results[i]["method"] = "SC Majority (Consistent)" if is_consistent else "SC Majority (Inconsistent)"

            # --- RAG 触发逻辑 ---
            if not is_consistent:
                rag_data_tuples.append((questions[i], i)) 
                rag_indices.append(i)

            # --- Baseline 赋值 ---
            if baseline_require:
                greedy_raw = greedy_outputs[i].outputs[0].text
                greedy_ans = extract_answer(greedy_raw)

                # Baseline 1: Direct Greedy
                results[i]["baseline_1"] = greedy_ans
                results[i]["baseline_1_raw"] = greedy_raw

                # Baseline 2: SC Majority (Always)
                results[i]["baseline_2"] = sc_outputs
                
                # Baseline 3: Inconsistent -> Greedy, Consistent -> SC
                if is_consistent:
                    results[i]["baseline_3"] = sc_outputs
                else:
                    results[i]["baseline_3"] = greedy_ans

        # [Phase 2] RAG Process (仅针对不一致的问题)
        if rag_data_tuples:
            print(f"\n🌀 [Phase 2.1] 对 {len(rag_data_tuples)} 道难题进行抽象化...")
            rag_q_texts = [item[0] for item in rag_data_tuples]
            
            # 1. 抽象化
            abstract_queries = self.student.batch_abstraction(rag_q_texts)
            
            print(f"\n🔍 [Phase 2.2] 检索经验...")
            # 2. 检索
            hints_list = self.db.batch_search(abstract_queries, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD)
            
            # === ⚡️ 核心修改开始 ⚡️ ===
            
            # 区分“有Hint的任务”和“无Hint的任务”
            real_rag_tasks = []      # 需要进 LLM 的 [(question, hint, original_index)]
            zero_hit_indices = []    # 没救的，直接复用 Greedy
            
            for i, hint in enumerate(hints_list):
                original_idx = rag_indices[i]
                if not hint:
                    # 😭 没检索到：直接复用 Baseline 1 的结果 (如果 Baseline 1 没跑，那没办法只能重跑，但通常你跑了)
                    zero_hit_indices.append(original_idx)
                else:
                    # 🤩 检索到了：加入重跑队列
                    real_rag_tasks.append((rag_q_texts[i], hint, original_idx))
            
            print(f"   -> 命中统计: {len(real_rag_tasks)} 题进入 RAG | {len(zero_hit_indices)} 题回退 Greedy (复用缓存)")

            # A. 处理 0 命中：直接回填 Baseline 1 的答案 (绝对对齐!)
            if baseline_require and greedy_outputs:
                for idx in zero_hit_indices:
                    # 这里的逻辑是：既然没有 Hint，RAG Prompt == Base Prompt
                    # 所以结果理论上等于 Greedy。为了消除 Batch 噪声，直接 Copy。
                    # 注意：这里假设 greedy_outputs[idx] 存在。
                    raw_text = greedy_outputs[idx].outputs[0].text
                    pred = extract_answer(raw_text)
                    
                    results[idx]['final_answer'] = pred
                    results[idx]['final_answer_raw'] = raw_text
                    results[idx]['method'] = "SC Inconsistent → Greedy (Fallback)"
                    results[idx]['retrieved_context'] = []
            else:
                # 如果没开 baseline_require，那这些 0 命中的题还得含泪重跑...
                # 但既然你在做 eval，通常都有 greedy_outputs
                pass 

            # B. 处理 有 命中：调用 LLM
            if real_rag_tasks:
                print(f"\n⚡️ [Phase 2.3] 运行 RAG 推理 (仅 {len(real_rag_tasks)} 题)...")
                real_qs = [t[0] for t in real_rag_tasks]
                real_hints = [t[1] for t in real_rag_tasks]
                real_indices = [t[2] for t in real_rag_tasks]
                
                # 这里调用你的 generate_rag (此时 hints 必定不为空)
                rag_outputs = self.student.generate_rag(real_qs, real_hints)
                
                for i, output in enumerate(rag_outputs):
                    original_idx = real_indices[i]
                    raw_text = output.outputs[0].text
                    pred = extract_answer(raw_text)
                    
                    results[original_idx]['final_answer'] = pred
                    results[original_idx]['final_answer_raw'] = raw_text
                    results[original_idx]['method'] = "SC Inconsistent → RAG"
                    results[original_idx]['retrieved_context'] = real_hints[i]

            # === ⚡️ 核心修改结束 ⚡️ ===

        return results
            # 初始化结果对象
            # res = {
            #     "question": questions[i],
            # }
            
            # 填充 Baseline 信息
        #     if baseline_require:
        #         # Baseline 2: Direct Greedy
        #         if greedy_outputs:
        #             res['baseline_2_greedy'] = extract_answer(greedy_raw)
        #             res['baseline_2_raw'] = greedy_outputs[i].outputs[0].text
                
        #         # Baseline 1: SC-3 Majority Vote (即使不一致也取众数)
        #         res['baseline_1_majority'] = maj_ans
        #         res['baseline_1_raw'] = raw_text

        #     if is_consistent and not force_rag:
        #         # SC一致，直接采纳
        #         res['prediction'] = maj_ans
        #         res['raw_output'] = raw_text
        #         res['method'] = "SC-3 (Consistent)"
        #         # 一致情况下，不需要 sc3_inconsistent_greedy 字段
        #         results[i] = res
        #     else:
        #         # SC不一致，进入RAG队列
        #         # 记录 SC-3 Inconsistent 时的 Greedy 结果（从 SC-3 中取第一个输出作为 greedy）
        #         sc3_greedy_raw = output.outputs[0].text
        #         sc3_greedy_pred = extract_answer(sc3_greedy_raw)
        #         res['sc3_inconsistent_greedy'] = sc3_greedy_pred
        #         res['sc3_inconsistent_greedy_raw'] = sc3_greedy_raw
                
        #         rag_indices.append(i)
        #         rag_questions.append(questions[i])
        #         results[i] = res # 先占位，后续更新 prediction
        
        # print(f"   -> 一致性通过: {consistent_count}/{len(questions)}")
        # print(f"   -> 需要 RAG 介入: {len(rag_questions)}/{len(questions)}")

        # # ================= Phase 2: Adaptive RAG =================
        # if rag_questions:
        #     print(f"\n🌀 [Phase 2.1] 对 {len(rag_questions)} 道难题进行抽象化...")
        #     abstract_queries = self.student.batch_abstraction(rag_questions)
            
        #     print(f"\n🔍 [Phase 2.2] 检索经验...")
        #     hints_list = self.db.batch_search(abstract_queries, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD)
            
        #     print(f"\n⚡️ [Phase 2.3] 运行 RAG 推理 (Greedy)...")
        #     rag_outputs = self.student.generate_rag(rag_questions, hints_list)
            
        #     for idx, rag_idx in enumerate(rag_indices):
        #         output = rag_outputs[idx]
        #         raw_text = output.outputs[0].text
        #         pred = extract_answer(raw_text) 
                
        #         # 更新 RAG 结果（SC-3 Inconsistent → RAG）
        #         results[rag_idx]['prediction'] = pred
        #         results[rag_idx]['raw_output'] = raw_text
        #         results[rag_idx]['method'] = "SC-3 Inconsistent → RAG"
        #         results[rag_idx]['retrieved_context'] = hints_list[idx]
        #         results[rag_idx]['retrieved_trigger'] = hints_list[idx][0]['trigger'] if hints_list[idx] else None
                
        # return results

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
