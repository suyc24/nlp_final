
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Universal_RAG.db_manager import MathNotebookDB

def inspect():
    db_path = "/root/autodl-tmp/nlp_final/Universal_RAG/math_notebook_db"
    print(f"Inspecting DB at: {db_path}")
    
    if not os.path.exists(db_path):
        print("❌ DB path does not exist!")
        return

    try:
        db = MathNotebookDB(db_path=db_path)
        count = db.collection.count()
        print(f"✅ Connected to DB.")
        print(f"📊 Collection 'elite_strategies' count: {count}")
        
        if count > 0:
            print("\n🔍 Peeking at top 3 items:")
            results = db.collection.get(limit=3)
            for i in range(len(results['ids'])):
                print(f"--- Item {i+1} ---")
                print(f"ID: {results['ids'][i]}")
                print(f"Document (Strategy): {results['documents'][i]}")
                print(f"Metadata: {results['metadatas'][i]}")
        else:
            print("⚠️ Collection is empty!")
            
    except Exception as e:
        print(f"❌ Error inspecting DB: {e}")

if __name__ == "__main__":
    inspect()
