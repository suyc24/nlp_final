
# ================= Teacher Prompts =================
TEACHER_SYSTEM_PROMPT = """You are a Meta-Cognitive Math Tutor specializing in "Knowledge Distillation". 
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

# ================= Student Prompts =================
def construct_base_prompt(question):
    return f"<|im_start|>user\nQuestion: {question}\nPlease reason step-by-step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"

def construct_hint_prompt(question, hint):
    return f"""<|im_start|>user
Hint from Tutor: {hint}

Question: {question}
Please reason step-by-step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

def construct_abstraction_prompt(q):
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

# def construct_rag_prompt(q, hints):
#     if not hints: return construct_base_prompt(q)
    
#     strategies_text = ""
#     for idx, h in enumerate(hints):
#         strategies_text += f"Strategy {idx+1} (Matched Scenario: {h['trigger']}):\n{h['strategy']}\n\n"
    
#     content = f"""Reference Knowledge:
# {strategies_text}
# ---
# [Demonstration of how to use the Strategy]
# Example Scenario:
# Reference Strategy: "To find the total distance, multiply the speed by the time."
# Question: A car travels at 60 mph for 3 hours. How far does it go?
# Reasoning: The Reference Strategy suggests multiplying speed by time. 
# Speed = 60, Time = 3. 
# Calculation: 60 * 3 = 180.
# The answer is \\boxed{{180}}.

# ---
# [Your Turn]
# Question: {q}
# Instruction: First, check if any of the "Reference Knowledge" above applies to this question. If yes, explicitly use that logic. Reason step-by-step, and put your final answer within \\boxed{{}}."""

#     return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

def construct_rag_prompt(q, hints):
    # 1. 必须保留的回退逻辑 (配合你 llm_client 的修改，双重保险)
    if not hints: 
        return construct_base_prompt(q)
    
    # 2. 极简拼接
    # 对于小模型，不要写 "Matched Scenario"，直接给干货(Strategy)
    # 使用列表符号 "-" 引导，清晰且节省 Token
    strategies_text = "\n".join([f"- {h['strategy']}" for h in hints])
    
    # 3. 构造 Content
    # 结构：[Tips] -> [Question] -> [Instruction]
    # 这里的 Instruction 和 Base Prompt 保持高度一致
    content = f"""Reference Tips:
{strategies_text}

Question: {q}
Please reason step-by-step, using the tips above if helpful, and put your final answer within \\boxed{{}}."""

    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"