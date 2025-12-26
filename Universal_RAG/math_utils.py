import re
from sympy import simplify, parse_expr
from func_timeout import func_timeout, FunctionTimedOut

# 尝试导入 latex2sympy2，如果不存在则回退
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
        """
        从文本中提取 \\boxed{...} 的内容。
        """
        if s is None: return None
        
        # ⚠️ 修复 1: 强制转换为字符串，防止输入是 float/int 导致报错
        s = str(s)
        
        if "\\boxed" not in s: 
            return None
            
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
        # 常用 LaTeX 符号清理
        replacements = [
            ("\\$", ""), ("\\text", ""), ("\\mbox", ""), ("\\mathrm", ""),
            ("\\,", ""), ("\\!", ""), ("\\ ", ""), ("%", ""),
            ("\\left", ""), ("\\right", ""), ("\\limits", ""), 
            ("°", "") 
        ]
        for old, new in replacements:
            s = s.replace(old, new)
        
        # 标准化分数和运算符
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        s = s.replace("\\div", "/")
        s = s.replace("\\cdot", "*")
        
        # 去除空白
        s = "".join(s.split())
        return s

    def _sympy_check_core(self, pred_str, gt_str):
        pred_sym = None
        gt_sym = None
        
        # 1. 尝试使用 latex2sympy
        if latex2sympy:
            try:
                pred_sym = latex2sympy(pred_str)
                gt_sym = latex2sympy(gt_str)
            except: pass

        # 2. 如果失败，尝试使用 SymPy 的 parse_expr (需做简单替换)
        if pred_sym is None or gt_sym is None:
            try:
                clean_pred = pred_str.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\frac", "")
                clean_gt = gt_str.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\frac", "")
                pred_sym = parse_expr(clean_pred)
                gt_sym = parse_expr(clean_gt)
            except: return False

        # 3. 比较差异是否为 0
        try:
            # simplify(a - b) == 0 比 a == b 更鲁棒
            return simplify(pred_sym - gt_sym) == 0
        except: return False

    def verify(self, model_output, ground_truth):
        """
        验证模型输出是否正确
        :param model_output: 模型生成的文本 (可以是完整文本，也可以是提取后的答案)
        :param ground_truth: 标准答案 (通常包含 \\boxed{})
        :return: (bool, extracted_answer)
        """
        # 1. 尝试提取 \\boxed{}
        pred_inner = self.remove_boxed(model_output)
        gt_inner = self.remove_boxed(ground_truth)
        
        # ⚠️ 修复 2: 兼容性处理
        # 如果 model_output 里没有 boxed (比如已经是提取好的 "12")，
        # 之前的代码会返回 None 导致直接判错。
        # 现在回退到使用原始字符串。
        if pred_inner is None: 
            pred_inner = str(model_output) if model_output is not None else ""
            
        if gt_inner is None: 
            gt_inner = str(ground_truth) if ground_truth is not None else ""

        # 如果为空，直接返回 False
        if not pred_inner or not gt_inner:
            return False, pred_inner

        norm_pred = self._clean_latex(pred_inner)
        norm_gt = self._clean_latex(gt_inner)

        # Level 1: String Match (最快)
        if norm_pred == norm_gt: return True, pred_inner
        
        # Level 2: Float Match (处理 3.5 vs 3.500)
        try:
            if abs(float(norm_pred) - float(norm_gt)) < 1e-4: return True, pred_inner
        except: pass
        
        # Level 3: Set Match (处理多解情况 "x=1, x=2" vs "x=2, x=1")
        if "," in norm_pred and "," in norm_gt:
            try:
                set_pred = sorted([self._clean_latex(x) for x in pred_inner.split(',') if x.strip()])
                set_gt = sorted([self._clean_latex(x) for x in gt_inner.split(',') if x.strip()])
                if set_pred == set_gt: return True, pred_inner
            except: pass
            
        # Level 4: SymPy Match (最慢但最强，处理代数等价)
        try:
            is_equiv = func_timeout(self.timeout, self._sympy_check_core, args=(pred_inner, gt_inner))
            if is_equiv: return True, pred_inner
        except: pass

        return False, pred_inner