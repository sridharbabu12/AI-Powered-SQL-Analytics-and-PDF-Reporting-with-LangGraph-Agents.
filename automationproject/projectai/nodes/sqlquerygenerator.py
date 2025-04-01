from projectai.state.state import State
from projectai.LLMS.supabase_client import supabase_client
from projectai.LLMS.claudellm import antropic_client
from dotenv import load_dotenv

load_dotenv()

def generate_sql_from_query(state:State):
    """
    Generate SQL query from the user's natural language query and schema information.
    
    Args:
        state: The state dictionary containing user query and schema info.
        
    Returns:
        A SQL query string.
    """
    # Debugging: Check the type of state
    print("State type:", type(state))  # Debugging line
    print("State value:", state)        # Debugging line

    if not isinstance(state, dict):
        raise ValueError("Expected state to be a dictionary.")

    # Create a detailed schema description with example values
    schema_description = "Database Schema:\n"
    
    for table, columns in state['schema_info'].items():
        schema_description += f"\nTable: {table}\n"
        # Get sample data for this table
        sample_query = f"""
        SELECT * FROM "{table}" LIMIT 1
        """
        try:
            sample_response = supabase_client.rpc('execute_sql', {'query': sample_query}).execute()
            sample_data = sample_response.data[0] if sample_response.data else {}
        except Exception:
            sample_data = {}
        
        for col in columns:
            col_name = col['column_name']
            col_type = col['data_type']
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            sample_value = sample_data.get(col_name, '')
            schema_description += f"  - Column: \"{col_name}\"\n"
            schema_description += f"    Type: {col_type}\n"
            schema_description += f"    Constraints: {nullable}\n"
            if sample_value:
                schema_description += f"    Example: {sample_value}\n"

    # Send the prompt to Claude
    message = antropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        temperature=0,
        system="""
        You are an SQL expert assisting with PostgreSQL queries for Supabase. Generate a syntactically correct SQL query based on the user's request.

        Guidelines:
        - In PostgreSQL, the ROUND function expects a numeric type. Always cast values to numeric before passing them to ROUND.
        - Use `ROUND(value::numeric, 2)` for rounding numbers to two decimal places.
        - Handle division by zero using `NULLIF` to avoid errors.
        - Ensure the query is a valid SELECT statement and does not include INSERT, UPDATE, DELETE, DROP, or ALTER.
        - Use double quotes ("column_name") for column names.
        - Only return the SQL query without any explanations, formatting, or additional text.
        - Limit results to 100 rows unless otherwise specified.
        - Ensure that the generated query logically follows the schema provided and uses only the columns that exist in the schema.
        """,
        messages=[
            {"role": "user", "content": f"""
            {schema_description}
            
            Generate a PostgreSQL query (without semicolon) for this request:
            {state['user_query']}
            
            Remember to:
            1. Only use columns that exist in the schema above.
            2. Use exact column names with proper case.
            3. Enclose column names containing spaces or special characters in double quotes.
            4. Do NOT prefix columns with table names unless necessary for joins.
            """}
        ]
    )
    
    # Extract and clean the SQL query
    try:
        sql_query = message.content[0].text.strip()
    except IndexError:
        raise ValueError("No SQL query returned from the model.")
    
    # Remove any trailing semicolons
    sql_query = sql_query.rstrip(';')
    
    # Log the generated query for debugging
    print(f"Generated SQL query: {sql_query}")
    
    # Check for unexpected characters in the SQL query
    if "{" in sql_query or "}" in sql_query:
        raise ValueError("Generated SQL query contains unexpected characters.")
    
    return {"sql_query": sql_query}