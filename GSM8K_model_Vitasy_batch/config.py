# config.py

# ================= 全局配置 =================
# 数据库路径 (默认用于训练和查看)
DB_PATH = "./math_notebook_db"

# ================= 训练配置 (prompt_intervention_lab.py) =================
# 训练使用的模型
STUDENT_MODEL_PATH = "Qwen/Qwen2.5-Math-0.5B"

# DeepSeek API 配置
DEEPSEEK_API_KEY = "" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 训练参数
SAMPLE_SIZE = 7000         # 增大样本量以发挥并发优势
MAX_RETRY_ROUNDS = 3       
TEACHER_CONCURRENCY = 15   # DeepSeek API 的并发线程数

# ================= 评测配置 (eval_rag_performance.py) =================
# 评测使用的模型 (通常与训练模型一致，或者是其Instruct版本)
EVAL_MODEL_PATH = STUDENT_MODEL_PATH

# 评测时使用的数据库路径 (默认与全局DB_PATH一致，可单独修改以测试不同库)
EVAL_DB_PATH = DB_PATH 

# 评测输出文件
OUTPUT_FILE = "final_rag_eval_abstracted.json"

# RAG 参数
TOP_K = 3               
SIMILARITY_THRESHOLD = 0.7  
BATCH_SIZE_EMBED = 64
SC_N = 3  # Self-Consistency 采样次数

# ================= 查看配置 (inspect_db.py) =================
# 查看时使用的数据库路径
INSPECT_DB_PATH = DB_PATH
COLLECTION_NAME = "elite_strategies"
EXPORT_FILE = "exported_experiences.json"
