
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ================= 全局配置 =================
# 数据库路径
DB_PATH = "./math_notebook_db"

# ================= 训练配置 =================
# 训练使用的模型
# 只要 HF_HOME 设置正确，可以直接使用模型名称
STUDENT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# 强制离线模式，防止联网检查更新

HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# DeepSeek API 配置 (Teacher)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 训练参数
SAMPLE_SIZE = 5000         
MAX_RETRY_ROUNDS = 3       
TEACHER_CONCURRENCY = 15   

# ================= 评测配置 =================
# 评测使用的模型
EVAL_MODEL_PATH = STUDENT_MODEL_PATH

# RAG 参数
TOP_K = 1               
SIMILARITY_THRESHOLD = 0.0
BATCH_SIZE_EMBED = 64
SC_N = 3  # Self-Consistency 采样次数

# HF Cache
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
