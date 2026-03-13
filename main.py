import time
import os
import sys

# Ensure modules can be imported if this script is run from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision import VisionModule
from voice import VoiceModule
from memory import MemoryModule
from brain import BrainModule

def main():
    print("Initializing Digital Brain...")
    
    # Initialize all modules
    # In a real app, you might lazily load these or handle errors if hardware is missing
    vision = VisionModule()
    voice = VoiceModule()
    memory = MemoryModule()
    brain = BrainModule()
    
    print("\n--- Digital Brain is Ready ---")
    
    try:
        while True:
            print("\n[System: Waiting for User Interaction...]")
            # In a real scenario, this might be triggered by a wake word or motion detection
            input("Press Enter to initiate recognition (or type 'quit' to exit)...")
            
            # 1. Capture Face
            print("\n[System: Capturing Face...]")
            frame = vision.capture_frame()
            face_emb = vision.generate_embedding(frame)
            if face_emb:
                print("[System: Face embedding generated successfully.]")
            else:
                print("[System: No face detected. Skipping face recognition.]")

            # 2. Capture Voice
            print("\n[System: Capturing Voice...]")
            audio = voice.capture_audio()
            voice_emb = voice.generate_embedding(audio)
            if voice_emb:
                print("[System: Voice embedding generated successfully.]")
            else:
                print("[System: Audio capture failed or too short. Skipping voice recognition.]")

            if not face_emb and not voice_emb:
                print("[System: Cannot proceed without multimodal inputs. Please try again.]")
                continue

            # 3. Match against Database
            print("\n[System: Querying Memory Database...]")
            matched_user_id = memory.match_user(face_embedding=face_emb, voice_embedding=voice_emb)
            
            is_new_user = False
            if matched_user_id:
                print(f"[System: *MATCH FOUND* - User ID: {matched_user_id}]")
            else:
                print("[System: *NO MATCH FOUND* - Auto-registering new user...]")
                matched_user_id = memory.register_new_user(face_embedding=face_emb, voice_embedding=voice_emb)
                is_new_user = True
                print(f"[System: *REGISTERED* - New User ID: {matched_user_id}]")

            # Interaction Loop with LLM
            profile = memory.get_user_profile(matched_user_id)
            history = memory.get_chat_history(matched_user_id)
            
            print(f"\n--- Interaction Started (User '{profile['name']}') ---")
            
            # Initial Greeting from Brain
            if is_new_user:
                greeting_prompt = "A new user just arrived and was auto-registered. Introduce yourself as Digital Brain and ask for their name so you can remember them."
                print("\n[Digital Brain is thinking...]")
                greeting = brain.generate_response(greeting_prompt, profile, history)
                print(f"\nDigital Brain: {greeting}")
                memory.add_chat_message(matched_user_id, "assistant", greeting)
            else:
                print("\n[Digital Brain is retrieving context...]")
                greeting_prompt = f"The user {profile['name']} just returned. Acknowledge their return warmly based on past context."
                greeting = brain.generate_response(greeting_prompt, profile, history)
                print(f"\nDigital Brain: {greeting}")
                memory.add_chat_message(matched_user_id, "assistant", greeting)

            # Chat Loop for this session
            while True:
                user_msg = input("\nYou: ")
                if user_msg.lower() in ['quit', 'exit', 'bye']:
                    print("Digital Brain: Goodbye!")
                    break # Breaks the inner chat loop, goes back to main waiting loop
                    
                # If it's a new user and they tell us their name (simple heuristic)
                if is_new_user and "my name is" in user_msg.lower():
                    # Extract name loosely (a real NLP extractor would be better)
                    words = user_msg.split()
                    name = words[-1].strip('.').capitalize()
                    memory.update_user_name(matched_user_id, name)
                    print(f"[System: Updated user name to {name}]")
                    profile = memory.get_user_profile(matched_user_id) # Refresh
                    
                memory.add_chat_message(matched_user_id, "user", user_msg)
                
                # Fetch updated history for the brain
                updated_history = memory.get_chat_history(matched_user_id)
                
                print("\n[Digital Brain is thinking...]")
                response = brain.generate_response(user_msg, profile, updated_history)
                
                print(f"\nDigital Brain: {response}")
                memory.add_chat_message(matched_user_id, "assistant", response)

            print("\n--- Interaction Ended ---")
            
    except KeyboardInterrupt:
        print("\nDigital Brain shutting down.")

if __name__ == "__main__":
    main()
