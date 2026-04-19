import chromadb
import os
import uuid
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, db_path="./data"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.db_path, "chroma_db"))
        
        self.face_collection = self.chroma_client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.voice_collection = self.chroma_client.get_or_create_collection(
            name="voice_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.knowledge_collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # --- Biometric Registry ---
    def match_user(self, face_embedding=None, voice_embedding=None, threshold=0.85):
        best_face_match = None
        best_voice_match = None
        distance_threshold = 1.0 - threshold

        if face_embedding is not None:
            results = self.face_collection.query(query_embeddings=[face_embedding], n_results=1)
            if results["ids"] and len(results["ids"][0]) > 0:
                distance = results["distances"][0][0]
                if distance <= distance_threshold:
                    best_face_match = results["ids"][0][0]

        if voice_embedding is not None:
            results = self.voice_collection.query(query_embeddings=[voice_embedding], n_results=1)
            if results["ids"] and len(results["ids"][0]) > 0:
                distance = results["distances"][0][0]
                if distance <= distance_threshold:
                    best_voice_match = results["ids"][0][0]

        if face_embedding is not None:
            return best_face_match
        else:
            return best_voice_match

    def register_new_user(self, face_embedding=None, voice_embedding=None):
        new_user_id = str(uuid.uuid4())
        if face_embedding is not None:
            self.face_collection.add(ids=[new_user_id], embeddings=[face_embedding])
        if voice_embedding is not None:
            self.voice_collection.add(ids=[new_user_id], embeddings=[voice_embedding])
        return new_user_id

    # --- Knowledge Base (RAG) ---
    def ingest_document(self, user_id: str, text: str, source_name: str, source_type: str = "document"):
        chunks = self.splitter.split_text(text)
        if not chunks: return

        embeddings = self.embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"user_id": user_id, "source": source_name, "type": source_type} for _ in chunks]
        
        self.knowledge_collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_context(self, user_id: str, query: str, top_k: int = 5) -> str:
        query_emb = self.embedder.encode([query]).tolist()
        results = self.knowledge_collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"user_id": user_id}
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        context_str = "\\n".join([f"- [{meta['source']}]: {doc}" 
                                   for doc, meta in zip(results['documents'][0], results['metadatas'][0])])
        return context_str

