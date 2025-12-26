
import os
import shutil
import time
import random
import chromadb
from sentence_transformers import SentenceTransformer
from . import config

class MathNotebookDB:
    def __init__(self, db_path=None, reset=False):
        self.db_path = db_path or config.DB_PATH
        if reset and os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            
        print(f"📚 db: {self.db_path}...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="elite_strategies")
        
        self.failed_collection = self.client.get_or_create_collection(name="temp_failed_cases")
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    def index_failed_cases(self, failed_cases):
        if not failed_cases: return
        print("index_failed_cases")
        
        documents = []
        for c in failed_cases:
            if 'abstract_question' in c and c['abstract_question']:
                documents.append(c['abstract_question'])
            else:
                documents.append(c['question'])

        ids = [str(c['id']) for c in failed_cases]
        embeddings = self.embedder.encode(documents).tolist()
        metadatas = [{"ground_truth": c['ground_truth']} for c in failed_cases]
        
        try:
            self.client.delete_collection("temp_failed_cases")
            self.failed_collection = self.client.create_collection("temp_failed_cases")
        except:
            pass
            
        self.failed_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search_similar_failed_case(self, trigger_text, exclude_id):
        embedding = self.embedder.encode(trigger_text).tolist()
        results = self.failed_collection.query(query_embeddings=[embedding], n_results=2)
        
        if not results['ids'][0]: return None
        
        for i, found_id in enumerate(results['ids'][0]):
            if str(found_id) != str(exclude_id):
                return {
                    "id": int(found_id),
                    "question": results['documents'][0][i],
                    "ground_truth": results['metadatas'][0][i]['ground_truth']
                }
        return None

    def save_experience_batch(self, experiences):
        if not experiences: return
        triggers = [e['trigger'] for e in experiences]
        embeddings = self.embedder.encode(triggers).tolist()
        ids = [f"exp_{int(time.time())}_{random.randint(10000,99999)}_{i}" for i in range(len(experiences))]
        documents = [e['rule_text'] for e in experiences]
        metadatas = [{"trigger": e['trigger'], "source_question": e['original_q'][:200]} for e in experiences]
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"save_experience: {len(experiences)}")

    def batch_search(self, queries, top_k=3, threshold=0.7):
        if not queries: return []
        
        q_embeddings = self.embedder.encode(
            queries, 
            batch_size=64, 
            show_progress_bar=False, 
            convert_to_numpy=True
        ).tolist()
        
        search_results = self.collection.query(
            query_embeddings=q_embeddings,
            n_results=top_k
        )
        
        retrieved_contexts = []
        for i in range(len(queries)):
            valid_hints = []
            if search_results['ids'][i]:
                for j in range(len(search_results['ids'][i])):
                    distance = search_results['distances'][i][j]
                    doc_text = search_results['documents'][i][j]
                    metadata = search_results['metadatas'][i][j]
                    if distance < threshold:
                        valid_hints.append({
                            "strategy": doc_text,
                            "trigger": metadata.get('trigger', 'Unknown'),
                            "score": 1 - distance
                        })
            retrieved_contexts.append(valid_hints)
        return retrieved_contexts
