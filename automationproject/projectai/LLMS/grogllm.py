import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# Ensure you have the correct import

class GroqLLM:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
    def get_llm_model(self):
        try:
            groq_api_key = self.groq_api_key
            selected_groq_model = 'qwen-2.5-32b'
            
            if not groq_api_key:  # Check if API key is empty or None
                raise ValueError("Please Enter the Groq API KEY")

            model = ChatGroq(api_key=groq_api_key, model=selected_groq_model)
            return model  # Ensure model is returned properly

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception: {e}")

# Invoke the model
llm_instance = GroqLLM()
model = llm_instance.get_llm_model()

        