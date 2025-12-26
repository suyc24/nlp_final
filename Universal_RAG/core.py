
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
        # Teacher is initialized on demand during training, or directly here
        self.teacher = DeepSeekTeacher()

    def train(self, training_data, verifier_func=None):
        """
        Execute dual verification distillation training process.
        """
        # 0. Initialize verifier
        if verifier_func is None:
            verifier_func = default_check_correct

        # 1. Data preprocessing
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

        print(f"🚀 Dual verification distillation starts (N={len(batch_data)})...")

        # ================= Phase 1: Batch base run to filter wrong cases =================
        print("\n[Phase 1] Batch base run (Base Run)...")
        base_answers = self.student.batch_solve(batch_data)
        
        active_failed_cases = []
        for i, ans in enumerate(base_answers):
            item = batch_data[i]
            if not verifier_func(ans, item['ground_truth']):
                item['wrong_ans'] = ans
                active_failed_cases.append(item)
        
        print(f"   -> Number of wrong cases: {len(active_failed_cases)} / {len(batch_data)}")

        # Index wrong cases for later generalization verification
        print("🧠 Abstracting wrong cases to build index...")
        failed_questions = [c['question'] for c in active_failed_cases]
        abstracted_questions = self.student.batch_abstraction(failed_questions)
        
        for i, c in enumerate(active_failed_cases):
            c['abstract_question'] = abstracted_questions[i]
            
        self.db.index_failed_cases(active_failed_cases)

        # ================= Phase 2: Iterative distillation =================
        for round_idx in range(1, config.MAX_RETRY_ROUNDS + 1):
            if not active_failed_cases: break
                
            print(f"\n[Round {round_idx}] Processing {len(active_failed_cases)} wrong cases...")

            # A. Teacher intervention
            teacher_results = self.teacher.batch_teach(active_failed_cases)
            
            ready_to_solve_cases = []
            for case in active_failed_cases:
                if case['id'] in teacher_results:
                    res = teacher_results[case['id']]
                    case['hint'] = res['strategy']
                    case['trigger'] = res['trigger'] 
                    ready_to_solve_cases.append(case)
            
            if not ready_to_solve_cases: break

            # B. Student redoes original question (Primary Verification)
            print(f"   ✍️ Student redoes original question...")
            new_answers = self.student.batch_solve(ready_to_solve_cases)

            # C. Select correctly redone questions, prepare for generalization verification
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

            # D. Generalization Verification
            final_success_buffer = []
            
            if candidates_for_generalization:
                print(f"   ⚔️ Generalization verification for {len(candidates_for_generalization)} experiences...")
                
                # 1. For each candidate, find similar wrong case (Neighbor)
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

                # 2. Batch generalization verification
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
            
            # E. Save experiences that passed dual verification
            if final_success_buffer:
                self.db.save_experience_batch(final_success_buffer)
            
            print(f"   -> Original question fixed: {len(candidates_for_generalization)} | Generalization passed: {len(final_success_buffer)}")
            active_failed_cases = still_failed_cases

        print("\n" + "="*50)
        print("🏆 Training finished")

    def predict(self, questions, force_rag=False, baseline_require=False):
        """
        Execute Adaptive RAG prediction process.
        """
        results = [{"question": question} for question in questions]

        # [Phase 0] Baseline 1: Greedy
        greedy_outputs = None
        if baseline_require:
            print(f"\n📉 [Phase 0] Run Direct Greedy Baseline (N={len(questions)})...")
            greedy_outputs = self.student.generate_greedy(questions)

        # [Phase 1] SC-N Majority Vote
        print(f"\n⚡️ [Phase 1] Run SC-{config.SC_N} voting (N={len(questions)})...")
        outputs_sc = self.student.generate_sc(questions)
        
        rag_data_tuples = [] # [(question_text, original_index), ...]
        rag_indices = []     # [original_index, ...]

        for i, output in enumerate(outputs_sc):
            sc_outputs, is_consistent, sc_raw_text = self._get_majority_vote(output)
            results[i]["is_consistent"] = is_consistent
            
            results[i]["final_answer"] = sc_outputs
            results[i]["final_answer_raw"] = sc_raw_text
            results[i]["method"] = "SC Majority (Consistent)" if is_consistent else "SC Majority (Inconsistent)"

            if not is_consistent:
                rag_data_tuples.append((questions[i], i)) 
                rag_indices.append(i)

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

        if rag_data_tuples:
            print(f"\n🌀 [Phase 2.1] {len(rag_data_tuples)} abstraction...")
            rag_q_texts = [item[0] for item in rag_data_tuples]
            
            abstract_queries = self.student.batch_abstraction(rag_q_texts)
            
            print(f"\n🔍 [Phase 2.2] search principles")
            hints_list = self.db.batch_search(abstract_queries, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD)
            
            real_rag_tasks = []     
            zero_hit_indices = []   
            
            for i, hint in enumerate(hints_list):
                original_idx = rag_indices[i]
                if not hint:
                    zero_hit_indices.append(original_idx)
                else:
                    real_rag_tasks.append((rag_q_texts[i], hint, original_idx))
            
            print(f"   -> hit: {len(real_rag_tasks)} for RAG | {len(zero_hit_indices)} back to Greedy")


            if baseline_require and greedy_outputs:
                for idx in zero_hit_indices:

                    raw_text = greedy_outputs[idx].outputs[0].text
                    pred = extract_answer(raw_text)
                    
                    results[idx]['final_answer'] = pred
                    results[idx]['final_answer_raw'] = raw_text
                    results[idx]['method'] = "SC Inconsistent → Greedy (Fallback)"
                    results[idx]['retrieved_context'] = []
            else:

                pass 


            if real_rag_tasks:
                print(f"\n⚡️ [Phase 2.3] Run RAG inference (only {len(real_rag_tasks)} questions)...")
                real_qs = [t[0] for t in real_rag_tasks]
                real_hints = [t[1] for t in real_rag_tasks]
                real_indices = [t[2] for t in real_rag_tasks]

                rag_outputs = self.student.generate_rag(real_qs, real_hints)
                
                for i, output in enumerate(rag_outputs):
                    original_idx = real_indices[i]
                    raw_text = output.outputs[0].text
                    pred = extract_answer(raw_text)
                    
                    results[original_idx]['final_answer'] = pred
                    results[original_idx]['final_answer_raw'] = raw_text
                    results[original_idx]['method'] = "SC Inconsistent → RAG"
                    results[original_idx]['retrieved_context'] = real_hints[i]


        return results

    def _get_majority_vote(self, output_obj):

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
        
        is_consistent = (count >= 2)
        
        best_raw_text = raw_texts[0]
        for i, val in enumerate(answers):
            if val == major_ans:
                best_raw_text = raw_texts[i]
                break
                
        return major_ans, is_consistent, best_raw_text
