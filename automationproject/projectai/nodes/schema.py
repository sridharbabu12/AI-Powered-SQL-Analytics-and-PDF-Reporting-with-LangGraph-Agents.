from typing import Dict, List, Any
from projectai.state.state import State
from projectai.LLMS.supabase_client import supabase_client
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables once at module level
load_dotenv()

async def get_schema_info(state:State):
    """
    Extract schema information from the Supabase database and create a Supabase client.
    
    Args:
        state: Application state object
        
    Returns:
        Dict containing table names as keys and column information as values
        
    Raises:
        HTTPException: If there's an error accessing the database
    """
    try:
        # Query to get table names
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        
        tables_response = supabase_client.rpc('execute_sql', {'query': tables_query}).execute()
        
        if not tables_response.data:
            return {}
            
        tables = [row['table_name'] for row in tables_response.data]
        
        schema_info = {}
        for table in tables:
            # Query to get column information
            columns_query = f"""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = '{table}'
            """
            
            columns_response = supabase_client.rpc('execute_sql', {'query': columns_query}).execute()
            
            if columns_response.data:
                schema_info[table] = columns_response.data
                
        return {"schema_info": schema_info}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch schema information: {str(e)}"
        )
