import sqlite3
import os
import chromadb
from chromadb.config import Settings
import uuid

class MemoryModule:
    """
    Handles both Vector Database (ChromaDB) for embeddings and 
    Relational Database (SQLite) for user context/history.
    """
    def __init__(self, db_path="./data"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
        # 1. Initialize SQLite Database
        self.sqlite_db_file = os.path.join(self.db_path, "user_memory.db")
        self._init_sqlite()

        # 2. Initialize ChromaDB for Vector Storage
        # ChromaDB runs locally and persists to the provided path
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.db_path, "chroma_db"))
        
        # We use cosine similarity (cosine space) for embeddings.
        # DeepFace and SpeechBrain embeddings generally work well with cosine similarity.
        self.face_collection = self.chroma_client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.voice_collection = self.chroma_client.get_or_create_collection(
            name="voice_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        print("MemoryModule (Vector DB & SQLite) initialized.")

    def _init_sqlite(self):
        """Creates the necessary tables in SQLite if they don't exist."""
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            # Table to store core user information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Table to store conversation history linked to a user_id
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    role TEXT, -- 'user' or 'system' or 'assistant'
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            conn.commit()

    # --- Vector DB Operations ---

    def match_user(self, face_embedding=None, voice_embedding=None, threshold=0.80):
        """
        Attempts to match the given embeddings against the database.
        ChromaDB returns distance. Since we use cosine distance, distance = 1 - cosine_similarity.
        Cosine distance of 0.25 is a similarity of 0.75, which is a good starting threshold.
        """
        best_face_match = None
        best_voice_match = None

        distance_threshold = 1.0 - threshold

        # Attempt to match Face
        if face_embedding is not None:
            results = self.face_collection.query(
                query_embeddings=[face_embedding],
                n_results=1
            )
            # Check if there's a result and if it meets confidence threshold
            if results["ids"] and len(results["ids"][0]) > 0:
                distance = results["distances"][0][0]
                if distance <= distance_threshold:
                    best_face_match = results["ids"][0][0] # user_id

        # Attempt to match Voice
        if voice_embedding is not None:
            results = self.voice_collection.query(
                query_embeddings=[voice_embedding],
                n_results=1
            )
            if results["ids"] and len(results["ids"][0]) > 0:
                distance = results["distances"][0][0]
                if distance <= distance_threshold:
                    best_voice_match = results["ids"][0][0] # user_id

        # Reconciliation Logic
        if face_embedding is not None:
            # If we have a visual face, we MUST rely strictly on it.
            # If the face doesn't match, it is definitively a new person.
            # We ignore random voice matches (like background fan noise or silence) if the face is new!
            return best_face_match
        else:
            # Only rely purely on voice if the camera is disabled or no face was detected
            return best_voice_match

    def register_new_user(self, face_embedding=None, voice_embedding=None):
        """
        Registers a new user, generating a new UUID and saving their embeddings.
        Returns the generated user_id.
        """
        new_user_id = str(uuid.uuid4())
        
        if face_embedding is not None:
            self.face_collection.add(
                ids=[new_user_id],
                embeddings=[face_embedding]
            )
            
        if voice_embedding is not None:
            self.voice_collection.add(
                ids=[new_user_id],
                embeddings=[voice_embedding]
            )

        # Insert placeholder record into SQLite
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            # The name is empty initially, the LLM will figure it out and update it
            cursor.execute('INSERT INTO users (user_id, name) VALUES (?, ?)', (new_user_id, "Unknown User"))
            conn.commit()
            
        return new_user_id

    # --- SQLite Database Operations ---

    def update_user_name(self, user_id, new_name):
        """Updates the user's name in SQLite."""
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET name = ? WHERE user_id = ?', (new_name, user_id))
            conn.commit()

    def get_user_profile(self, user_id):
        """Retrieves user info."""
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return {"user_id": user_id, "name": result[0]}
            return None

    def add_chat_message(self, user_id, role, message):
        """Appends a message to the user's chat history."""
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)',
                (user_id, role, message)
            )
            conn.commit()

    def get_chat_history(self, user_id, limit=10):
        """Retrieves recent chat history for context injection into the LLM."""
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT role, message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
                (user_id, limit)
            )
            results = cursor.fetchall()
            # Return ordered chronologically (oldest first)
            return results[::-1]

# Simple test block
if __name__ == "__main__":
    memory = MemoryModule()
    
    # Mock embeddings (e.g., 512 dimensions for FaceNet, 192 for SpeechBrain)
    mock_face_emb = [0.1] * 512
    mock_voice_emb = [0.2] * 192
    
    # 1. Try to match (should fail)
    matched_id = memory.match_user(face_embedding=mock_face_emb)
    print(f"Initial Match attempt ID: {matched_id}")
    
    if not matched_id:
        # 2. Register
        new_id = memory.register_new_user(face_embedding=mock_face_emb, voice_embedding=mock_voice_emb)
        print(f"Registered new user with ID: {new_id}")
        
        # 3. Add History
        memory.add_chat_message(new_id, "user", "Hi, my name is Alex.")
        memory.update_user_name(new_id, "Alex")
        
        # 4. Try Match again
        matched_id_2 = memory.match_user(face_embedding=mock_face_emb)
        print(f"Second Match attempt ID: {matched_id_2}")
        
        if matched_id_2:
            profile = memory.get_user_profile(matched_id_2)
            history = memory.get_chat_history(matched_id_2)
            print(f"Profile: {profile}")
            print(f"History: {history}")
