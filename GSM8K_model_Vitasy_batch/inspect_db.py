import chromadb
import json
import os
import pandas as pd
from datetime import datetime
import config

# ================= 配置 =================
DB_PATH = config.INSPECT_DB_PATH
COLLECTION_NAME = config.COLLECTION_NAME
EXPORT_FILE = config.EXPORT_FILE

class NotebookInspector:
    def __init__(self):
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"❌ 找不到数据库路径: {DB_PATH}")
            
        print(f"📂 正在连接数据库: {DB_PATH}...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"✅ 成功加载集合: {COLLECTION_NAME}")
        except Exception as e:
            print(f"❌ 无法加载集合 '{COLLECTION_NAME}'。可能是名字不对或库为空。")
            print(f"   错误信息: {e}")
            # 列出所有可用集合
            col_list = self.client.list_collections()
            print(f"   现有集合列表: {[c.name for c in col_list]}")
            exit(1)

    def fetch_all(self):
        """拉取所有数据"""
        # include 参数指定要获取哪些字段
        data = self.collection.get(include=['metadatas', 'documents', 'embeddings'])
        count = len(data['ids'])
        print(f"📊 当前库存经验总数: {count} 条")
        return data

    def display_samples(self, num_samples=5):
        """在终端打印前 N 条样本"""
        data = self.fetch_all()
        count = len(data['ids'])
        
        print(f"\n=== 随机预览 (前 {min(count, num_samples)} 条) ===")
        
        for i in range(min(count, num_samples)):
            doc = data['documents'][i]
            meta = data['metadatas'][i]
            id_ = data['ids'][i]
            
            print(f"\n[ID]: {id_}")
            print(f"📌 Trigger (适用场景): {meta.get('trigger', 'N/A')}")
            print(f"💡 Strategy (策略): \n{doc}")
            print(f"🔗 Source Question (来源题片段): {meta.get('source_question', 'N/A')[:100]}...")
            print("-" * 60)

    def export_to_json(self):
        """导出所有数据到 JSON"""
        data = self.fetch_all()
        count = len(data['ids'])
        
        export_list = []
        for i in range(count):
            item = {
                "id": data['ids'][i],
                "trigger": data['metadatas'][i].get('trigger'),
                "strategy": data['documents'][i],
                "source_question": data['metadatas'][i].get('source_question')
            }
            export_list.append(item)
            
        with open(EXPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(export_list, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 已将所有 {count} 条经验导出至: {os.path.abspath(EXPORT_FILE)}")
        
    def search_by_keyword(self, keyword):
        """简单的关键词搜索（非向量搜索，仅文本匹配）"""
        print(f"\n🔍 正在搜索关键词: '{keyword}' ...")
        data = self.fetch_all()
        found_count = 0
        
        for i in range(len(data['ids'])):
            doc = data['documents'][i]
            meta = data['metadatas'][i]
            trigger = meta.get('trigger', '')
            
            # 在 Trigger 或 Strategy 中搜索
            if keyword.lower() in doc.lower() or keyword.lower() in trigger.lower():
                print(f"   Found in [{data['ids'][i]}]: Trigger='{trigger}'")
                found_count += 1
                
        if found_count == 0:
            print("   未找到相关经验。")

if __name__ == "__main__":
    inspector = NotebookInspector()
    
    # 1. 在终端显示前 5 条
    inspector.display_samples(5)
    
    # 2. 导出所有数据
    inspector.export_to_json()