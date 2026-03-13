import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv(override=True)

class BrainModule:
    """
    Handles LLM Orhcestration using LangChain. 
    It takes user history and context to generate personalized responses.
    """
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.7):
        # Requires GOOGLE_API_KEY environment variable to be set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY environment variable not set. LLM calls will fail.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        # Define the system prompt template
        self.system_prompt = """
        You are "Digital Brain", a highly intelligent, personalized AI assistant.
        Your goal is to be helpful, concise, and friendly.
        
        {user_context}
        
        When responding, consider the user's past interactions and preferences. 
        If they refer to past conversations, use the provided history to answer.
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the LangChain run chain
        self.chain = self.prompt_template | self.llm
        print("BrainModule (LLM) initialized.")

    def format_history(self, raw_history):
        """
        Converts the raw history from SQLite into LangChain message objects.
        raw_history format: [(role, message), (role, message), ...]
        """
        langchain_history = []
        for role, message in raw_history:
            if role == "user":
                langchain_history.append(HumanMessage(content=message))
            elif role in ["assistant", "system", "ai"]:
                langchain_history.append(AIMessage(content=message))
        return langchain_history

    def generate_response(self, user_input, user_profile, chat_history):
        """
        Generates a response from the LLM given the input and context.
        
        Args:
            user_input: The text the user just said/typed.
            user_profile: Dict containing user info, e.g. {"user_id": "...", "name": "..."}
            chat_history: List of tuples from the DB [(role, msg), ...]
            
        Returns:
            The text response from the LLM.
        """
        
        # Determine user context for the prompt
        if user_profile and user_profile.get("name") and user_profile.get("name") != "Unknown User":
            user_context = f"The user you are speaking to is named {user_profile['name']}."
            # If we know their name, it's a returning user
            user_context += " They are a returning user. Welcome them back warmly if they just arrived."
        else:
            user_context = "This is a new user who has not provided their name yet. \
            Politely ask for their name so you can remember them."

        # Format history for LangChain
        formatted_history = self.format_history(chat_history)
        
        try:
            # Gemini strictly requires the final message to be from a human. 
            # If the user input is empty (e.g. initial greeting), provide a silent prompt.
            if not user_input.strip():
                user_input = "Hello! Please greet me."
                
            # Invoke the chain
            response = self.chain.invoke({
                "user_context": user_context,
                "history": formatted_history,
                "input": user_input
            })
            return response.content
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "I'm sorry, I'm having trouble connecting to my brain right now."

# Simple test block
if __name__ == "__main__":
    # Note: This test requires a valid GOOGLE_API_KEY environment variable
    # export GOOGLE_API_KEY="AIza..."
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
        print(f"Test failed (make sure API key is set): {e}")
