import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class BrainModule:
    """
    Handles LLM Orchestration using the OpenAI Python client over Nvidia's API.
    It takes user history and context to generate personalized responses.
    """
    def __init__(self, model_name="meta/llama-3.1-8b-instruct", temperature=0.7):
        # We use the provided Nvidia API endpoint and key
        api_key = "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"
        
        self.model_name = model_name
        self.temperature = temperature
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=10.0
        )
        
        # Define the system prompt template
        self.system_prompt_template = """
        You are "Digital Brain", a highly intelligent, personalized AI assistant.
        Your goal is to be helpful, concise, and friendly.
        
        {user_context}
        
        When responding, consider the user's past interactions and preferences. 
        If they refer to past conversations, use the provided history to answer.
        """
        print("BrainModule (Nvidia Phi-4) initialized.")

    def format_history(self, raw_history):
        """
        Converts the raw history from SQLite into OpenAI dict objects.
        raw_history format: [(role, message), (role, message), ...]
        """
        openai_history = []
        for role, message in raw_history:
            if role == "user":
                openai_history.append({"role": "user", "content": message})
            elif role in ["assistant", "system", "ai"]:
                openai_history.append({"role": "assistant", "content": message})
        return openai_history

    def generate_response(self, user_input, user_profile, chat_history):
        """
        Generates a response from the LLM given the input and context.
        """
        
        # Determine user context for the prompt
        if user_profile and user_profile.get("name") and user_profile.get("name") != "Unknown User":
            user_context = f"The user you are speaking to is named {user_profile['name']}."
            # If we know their name, it's a returning user
            user_context += " They are a returning user. Welcome them back warmly if they just arrived."
        else:
            user_context = "This is a new user who has not provided their name yet. \
            Politely ask for their name so you can remember them."

        system_prompt = self.system_prompt_template.replace("{user_context}", user_context)

        # Format history for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.format_history(chat_history))
        
        try:
            # If the user input is empty (e.g. initial greeting), provide a silent prompt.
            if not user_input.strip():
                user_input = "Hello! Please greet me."
                
            messages.append({"role": "user", "content": user_input})
                
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                stream=True
            )
            
            # NVIDIA API heavily prefers streaming to prevent proxy timeouts
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    
            return full_response
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "I'm sorry, I'm having trouble connecting to my brain right now."

# Simple test block
if __name__ == "__main__":
    brain = BrainModule()
    
    mock_profile = {"user_id": "123", "name": "Sarah"}
    mock_history = [
        ("user", "My favorite color is blue."),
        ("assistant", "Noted! Your favorite color is blue.")
    ]
    
    user_input = "Do you remember my favorite color?"
    
    print(f"Testing with input: '{user_input}'")
    try:
        response = brain.generate_response(user_input, mock_profile, mock_history)
        print(f"\nLLM Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")
