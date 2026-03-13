import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision import VisionModule
from voice import VoiceModule
from memory import MemoryModule
from brain import BrainModule

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Allow up to 50MB payloads
CORS(app) # Enable CORS for all routes so React can communicate with it

# Initialize Modules
print("Initializing modules... (This may take a moment)")
vision = VisionModule()
voice = VoiceModule()
memory = MemoryModule()
brain = BrainModule()
print("Modules initialized.")

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "running"}), 200

@app.route('/api/identify', methods=['POST'])
def identify():
    """
    Accepts a JSON payload containing base64 encoded image and audio strings.
    Extracts embeddings, matches against ChromaDB, registers if new, and returns a greeting.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400
        
    image_b64 = data.get('image')
    audio_b64 = data.get('audio')
    
    # 1. Generate Face Embeddings
    face_emb = None
    if image_b64:
        try:
            faces = vision.generate_embeddings_from_base64(image_b64)
            if faces:
                # For manual identity endpoint, just take the first face
                face_emb = faces[0]["embedding"]
        except Exception as e:
            print(f"Face exception: {e}")
    
    # 2. Generate Voice Embedding
    voice_emb = None
    if audio_b64:
        try:
            voice_emb = voice.generate_embedding_from_base64(audio_b64)
        except Exception as e:
            print(f"Voice exception: {e}")
        
    if not face_emb and not voice_emb:
        print("Both Face and Voice embeddings returned None.")
        return jsonify({"error": "Could not detect face or voice from the provided media. Please ensure your face is visible."}), 400
        
    # 3. Match user embeddings in ChromaDB
    matched_user_id = memory.match_user(face_embedding=face_emb, voice_embedding=voice_emb)
    
    is_new = False
    if not matched_user_id:
        # Register new
        matched_user_id = memory.register_new_user(face_embedding=face_emb, voice_embedding=voice_emb)
        is_new = True
        status_msg = "Registered New User"
    else:
        status_msg = "Recognized Existing User"
        
    profile = memory.get_user_profile(matched_user_id)
    history = memory.get_chat_history(matched_user_id)
    
    # 4. Generate initial greeting via Gemini
    if is_new:
        prompt = "A new user just arrived and was auto-registered. Introduce yourself as Digital Brain and politely ask for their name so you can remember them."
    else:
        prompt = f"The user {profile['name']} just returned. Acknowledge their return warmly based on past context."
        
    greeting = brain.generate_response(prompt, profile, history)
    memory.add_chat_message(matched_user_id, "assistant", greeting)
    
    return jsonify({
        "user_id": matched_user_id,
        "profile": profile,
        "status": status_msg,
        "greeting": greeting
    })


@app.route('/api/poll_vision', methods=['POST'])
def poll_vision():
    """
    Receives continuous video frames from the React frontend.
    Extracts all faces, matches/registers them in ChromaDB, and returns bounding boxes + names.
    This works without audio.
    """
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
            
            # Match
            matched_user_id = memory.match_user(face_embedding=emb)
            
            is_new = False
            if not matched_user_id:
                # Auto register silent background detections
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
    """
    Accepts text messages from the user and returns the LLM response.
    """
    data = request.json
    user_id = data.get('user_id')
    user_msg = data.get('message')
    
    if not user_id or not user_msg:
        return jsonify({"error": "Missing user_id or message"}), 400
        
    # Check for name update logic (simple heuristic algorithm)
    profile = memory.get_user_profile(user_id)
    if profile['name'] == "Unknown User" and "my name is" in user_msg.lower():
        words = user_msg.split()
        name = words[-1].strip('.').capitalize()
        memory.update_user_name(user_id, name)
        profile = memory.get_user_profile(user_id)
        print(f"[System] Updated user name to {name}")
        
    memory.add_chat_message(user_id, "user", user_msg)
    history = memory.get_chat_history(user_id)
    
    # Generate contextual LLM response
    response = brain.generate_response(user_msg, profile, history)
    memory.add_chat_message(user_id, "assistant", response)
    
    return jsonify({
        "response": response,
        "profile": profile
    })

if __name__ == '__main__':
    print("Starting Digital Brain Headless API on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
