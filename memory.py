import os
from relational_db import RelationalDB
from vector_db import VectorDB

class MemoryModule:
    """
    Orchestrates both Vector Database (ChromaDB) for embeddings/RAG and 
    Relational Database (SQLite) for user context/history.
    """
    def __init__(self, db_path="./data"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
        self.rdb = RelationalDB(db_path=db_path)
        self.vdb = VectorDB(db_path=db_path)

        print("MemoryModule (Vector DB & SQLite via abstractions) initialized.")

    # --- Biometric & User Registry ---
    def match_user(self, face_embedding=None, voice_embedding=None, threshold=0.85):
        return self.vdb.match_user(face_embedding, voice_embedding, threshold)

    def register_new_user(self, face_embedding=None, voice_embedding=None):
        new_user_id = self.vdb.register_new_user(face_embedding, voice_embedding)
        self.rdb.insert_user(new_user_id)
        return new_user_id

    # --- Profile & Facts ---
    def update_user_name(self, user_id, new_name):
        self.rdb.update_user_name(user_id, new_name)

    def get_user_profile(self, user_id):
        profile = self.rdb.get_user_profile(user_id)
        if profile:
            profile['facts'] = self.rdb.get_persistent_facts(user_id)
        return profile
    
    def save_persistent_facts(self, user_id, facts_dict):
        import json
        self.rdb.save_persistent_facts(user_id, json.dumps(facts_dict))

    # --- Episodic Chat History ---
    def add_chat_message(self, user_id, role, message):
        self.rdb.add_chat_message(user_id, role, message)

    def get_chat_history(self, user_id, limit=10):
        return self.rdb.get_chat_history(user_id, limit)

    # --- Knowledge Base (RAG) ---
    def ingest_document(self, user_id: str, text: str, source_name: str, source_type: str = "document"):
        self.vdb.ingest_document(user_id, text, source_name, source_type)

    def retrieve_context(self, user_id: str, query: str, top_k: int = 5) -> str:
        return self.vdb.retrieve_context(user_id, query, top_k)
