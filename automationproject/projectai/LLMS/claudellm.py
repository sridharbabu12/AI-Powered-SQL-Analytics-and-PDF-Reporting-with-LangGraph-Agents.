from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

antropic_api_key=os.getenv("ANTHROPIC_API_KEY")

antropic_client=anthropic.Anthropic(api_key=antropic_api_key)
