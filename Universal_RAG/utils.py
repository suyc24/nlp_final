
import re

def extract_answer(text):
    if not text: return None
    text = text.replace(',', '')
    match = re.search(r'\\boxed\{(\-?\d+\.?\d*)\}', text)
    if match: return float(match.group(1))
    matches = re.findall(r'-?\d+\.?\d*', text[-100:])
    if matches: return float(matches[-1])
    return None

def default_check_correct(pred, gt):
    """
    默认的 GSM8K 风格验证器 (数值比较)
    """
    if "####" in gt: gold = extract_answer(gt.split("####")[1])
    else: gold = extract_answer(gt)
    val = extract_answer(pred)
    if gold is None or val is None: return False
    return abs(gold - val) < 1e-4
