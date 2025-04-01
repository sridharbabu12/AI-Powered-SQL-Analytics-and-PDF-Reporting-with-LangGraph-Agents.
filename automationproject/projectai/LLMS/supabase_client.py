import supabase
from dotenv import load_dotenv
import os

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
    
if not supabase_url or not supabase_key:
    raise ValueError("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")
    
supabase_client = supabase.create_client(supabase_url, supabase_key)

            