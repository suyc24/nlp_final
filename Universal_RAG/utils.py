
import re

def extract_answer(text):
    if text is None: return None
    # If already numeric, return as float
    if isinstance(text, (int, float)):
        return float(text)
    # Coerce to string for further processing
    text = str(text)
    if not text: return None
    text = text.replace(',', '')
    match = re.search(r'\\boxed\{(\-?\d+\.?\d*)\}', text)
    if match: return float(match.group(1))
    matches = re.findall(r'-?\d+\.?\d*', text[-100:])
    if matches: return float(matches[-1])
    return None

def default_check_correct(pred, gt):
    if isinstance(gt, str) and "####" in gt: gold = extract_answer(gt.split("####")[1])
    else: gold = extract_answer(gt)
    val = extract_answer(pred)
    if gold is None or val is None: return False
    return abs(gold - val) < 1e-4
