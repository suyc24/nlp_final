from openai import OpenAI
import os
import time

# 填入你的 Key
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("📡 正在尝试连接 DeepSeek API...")
start = time.time()
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello, are you online? Reply 1 word."}],
        max_tokens=10,
        timeout=10 # 设置短超时
    )
    print(f"✅ 连接成功! 耗时: {time.time()-start:.2f}s")
    print(f"回复: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 连接失败: {e}")