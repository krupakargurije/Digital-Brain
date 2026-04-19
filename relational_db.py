import sqlite3
import os

class RelationalDB:
    def __init__(self, db_path="./data"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.sqlite_db_file = os.path.join(self.db_path, "user_memory.db")
        self._init_sqlite()

    def _init_sqlite(self):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Add persistent_facts column if it doesn't exist (migrations)
            try:
                cursor.execute('ALTER TABLE users ADD COLUMN persistent_facts TEXT DEFAULT "{}"')
            except sqlite3.OperationalError:
                pass # Column exists

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    role TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            conn.commit()

    def update_user_name(self, user_id, new_name):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET name = ? WHERE user_id = ?', (new_name, user_id))
            conn.commit()

    def get_user_profile(self, user_id):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return {"user_id": user_id, "name": result[0]}
            return None
            
    def save_persistent_facts(self, user_id, facts_json_str):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET persistent_facts = ? WHERE user_id = ?', (facts_json_str, user_id))
            conn.commit()
            
    def get_persistent_facts(self, user_id):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT persistent_facts FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result and result[0]:
                import json
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    return {}
            return {}

    def insert_user(self, user_id, name="Unknown User"):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR IGNORE INTO users (user_id, name) VALUES (?, ?)', (user_id, name))
            conn.commit()

    def add_chat_message(self, user_id, role, message):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)',
                (user_id, role, message)
            )
            conn.commit()

    def get_chat_history(self, user_id, limit=10):
        with sqlite3.connect(self.sqlite_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT role, message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
                (user_id, limit)
            )
            results = cursor.fetchall()
            return results[::-1]

