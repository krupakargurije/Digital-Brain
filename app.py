import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision import VisionModule
from voice import VoiceModule
from memory import MemoryModule
from brain import BrainModule

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  
CORS(app) 

# Initialize Modules
print("Initializing modules... (This may take a moment)")
vision = VisionModule()
voice = VoiceModule()
memory = MemoryModule()
brain = BrainModule()
print("Modules initialized.")

executor = ThreadPoolExecutor(max_workers=4)

def async_extract_facts(user_id, user_msg, current_facts):
    """Background task to extract facts safely."""
    prompt = f"""
    Analyze the user's message and determine if it contains a permanent fact, such as their name, profession, or a preference.
    Current known facts: {current_facts}
    Message: "{user_msg}"
    Output valid JSON ONLY in this format: {{"updated_facts": {{"key1": "value1", "name": "User Name"}}}}
    If there are no new facts, just output an empty dictionary for updated_facts.
    """
    response = brain.generate_direct(prompt)
    try:
        # Strip markdown json block tags if the LLM adds them
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        new_facts = data.get("updated_facts", {})
        if new_facts:
            # Merge facts
            current_facts.update(new_facts)
            memory.save_persistent_facts(user_id, current_facts)
            if "name" in new_facts:
                memory.update_user_name(user_id, new_facts["name"])
            print(f"[System] Updated persistent facts: {new_facts}")
    except Exception as e:
        print(f"[System] Fact extraction failed or returned invalid JSON. Error: {e}, Response: {response}")

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "running"}), 200

@app.route('/api/identify', methods=['POST'])
def identify():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400
        
    image_b64 = data.get('image')
    audio_b64 = data.get('audio')
    
    # 1 & 2. Run vision and voice extraction concurrently
    future_face = executor.submit(vision.generate_embeddings_from_base64, image_b64) if image_b64 else None
    future_voice = executor.submit(voice.generate_embedding_from_base64, audio_b64) if audio_b64 else None
    
    face_emb = None
    voice_emb = None
    
    if future_face:
        try:
            faces = future_face.result(timeout=10)
            if faces: face_emb = faces[0]["embedding"]
        except Exception as e:
            print(f"Face extraction failed: {e}")
            
    if future_voice:
        try:
            voice_emb = future_voice.result(timeout=10)
        except Exception as e:
            print(f"Voice extraction failed: {e}")
        
    if not face_emb and not voice_emb:
        return jsonify({"error": "Could not detect face or voice from the provided media."}), 400
        
    # 3. Match user embeddings in ChromaDB
    matched_user_id = memory.match_user(face_embedding=face_emb, voice_embedding=voice_emb)
    
    is_new = False
    if not matched_user_id:
        matched_user_id = memory.register_new_user(face_embedding=face_emb, voice_embedding=voice_emb)
        is_new = True
        status_msg = "Registered New User"
    else:
        status_msg = "Recognized Existing User"
        
    profile = memory.get_user_profile(matched_user_id)
    if not profile:
        profile = {"user_id": matched_user_id, "name": "Unknown User", "facts": {}}
    history = memory.get_chat_history(matched_user_id)
    
    # 4. Generate initial greeting
    if is_new:
        prompt = "A new user just arrived and was auto-registered. Introduce yourself as Digital Brain and politely ask for their name or facts to remember."
    else:
        prompt = "The user just returned. Acknowledge their return warmly based on past context."
        
    # Retrieve RAG context
    rag_context = memory.retrieve_context(matched_user_id, prompt)
        
    greeting = brain.generate_response(prompt, profile, history, rag_context)
    if not greeting.startswith("I'm sorry, I'm having trouble"):
        memory.add_chat_message(matched_user_id, "assistant", greeting)
    
    return jsonify({
        "user_id": matched_user_id,
        "profile": profile,
        "status": status_msg,
        "greeting": greeting
    })

@app.route('/api/poll_vision', methods=['POST'])
def poll_vision():
    data = request.json
    if not data or not data.get('image'):
        return jsonify({"error": "No image provided."}), 400
        
    image_b64 = data.get('image')
    faces_data = []
    
    try:
        faces = vision.generate_embeddings_from_base64(image_b64)
        for face in faces:
            emb = face["embedding"]
            box = face["box"]
            matched_user_id = memory.match_user(face_embedding=emb)
            is_new = False
            if not matched_user_id:
                matched_user_id = memory.register_new_user(face_embedding=emb)
                is_new = True
                
            profile = memory.get_user_profile(matched_user_id)
            faces_data.append({
                "user_id": matched_user_id,
                "name": profile["name"] if profile else "Unknown User",
                "box": box,
                "is_new": is_new
            })
    except Exception as e:
        print(f"Poll vision exception: {e}")
        return jsonify({"error": str(e)}), 500
        
    return jsonify({"faces": faces_data})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    user_msg = data.get('message')
    
    if not user_id or not user_msg:
        return jsonify({"error": "Missing user_id or message"}), 400
        
    profile = memory.get_user_profile(user_id)
    if not profile:
        profile = {"user_id": user_id, "name": "Unknown User", "facts": {}}
        
    # Async facts extraction to avoid blocking response
    executor.submit(async_extract_facts, user_id, user_msg, profile.get("facts", {}))
        
    memory.add_chat_message(user_id, "user", user_msg)
    history = memory.get_chat_history(user_id)
    
    # RAG Retrieval
    rag_context = memory.retrieve_context(user_id, user_msg)
    
    # Generate contextual LLM response
    response = brain.generate_response(user_msg, profile, history, rag_context)
    
    if not response.startswith("I'm sorry, I'm having trouble"):
        memory.add_chat_message(user_id, "assistant", response)
        
    # We update profile if facts changed, though UI might just use response.
    return jsonify({
        "response": response,
        "profile": profile
    })

@app.route('/api/knowledge/upload', methods=['POST'])
def upload_knowledge():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    user_id = request.form.get('user_id')
    
    if not user_id:
        return jsonify({"error": "Missing user_id form data"}), 400
        
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF: {e}"}), 500
    elif file.filename.endswith(".txt"):
        text = file.read().decode('utf-8')
    else:
        return jsonify({"error": "Unsupported file type. Use .pdf or .txt"}), 400
        
    if text.strip():
        # Offload to background so we don't freeze the request on embeddings
        executor.submit(memory.ingest_document, user_id, text, file.filename, "document")
        return jsonify({"status": "Success", "message": f"Ingesting {file.filename} in the background."})
    return jsonify({"error": "File was empty or unreadable"}), 400

@app.route('/api/knowledge/note', methods=['POST'])
def add_note():
    data = request.json
    user_id = data.get('user_id')
    note = data.get('note')
    
    if not user_id or not note:
        return jsonify({"error": "Missing user_id or note"}), 400
        
    executor.submit(memory.ingest_document, user_id, note, "User Note", "note")
    return jsonify({"status": "Success", "message": "Note ingested."})

if __name__ == '__main__':
    print("Starting Digital Brain Headless API on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
