
import re
from sympy import simplify, parse_expr
from func_timeout import func_timeout, FunctionTimedOut
try:
    from latex2sympy2 import latex2sympy
except ImportError:
    latex2sympy = None

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
        验证模型输出是否正确
        :param model_output: 模型生成的完整文本 (包含 \\boxed{})
        :param ground_truth: 标准答案 (包含 \\boxed{})
        :return: (bool, extracted_answer)
        """
        pred_inner = self.remove_boxed(model_output)
        gt_inner = self.remove_boxed(ground_truth)
        
        if gt_inner is None: gt_inner = ground_truth
        if pred_inner is None: return False, None

        norm_pred = self._clean_latex(pred_inner)
        norm_gt = self._clean_latex(gt_inner)

        # Level 1: String Match
        if norm_pred == norm_gt: return True, pred_inner
        
        # Level 2: Float Match
        try:
            if abs(float(norm_pred) - float(norm_gt)) < 1e-4: return True, pred_inner
        except: pass
        
        # Level 3: Set Match (for multiple answers like "x=1, x=2")
        if "," in norm_pred and "," in norm_gt:
            try:
                set_pred = sorted([self._clean_latex(x) for x in pred_inner.split(',') if x.strip()])
                set_gt = sorted([self._clean_latex(x) for x in gt_inner.split(',') if x.strip()])
                if set_pred == set_gt: return True, pred_inner
            except: pass
            
        # Level 4: SymPy Match
        try:
            is_equiv = func_timeout(self.timeout, self._sympy_check_core, args=(pred_inner, gt_inner))
            if is_equiv: return True, pred_inner
        except: pass

        return False, pred_inner
