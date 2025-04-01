import pandas as pd
from projectai.LLMS.supabase_client import supabase_client
from projectai.state.state import State


        
def query_database(state: State):
    """
    Executes a SQL query using Supabase RPC and returns the result as a Pandas DataFrame.

    Args:
        state (dict): A dictionary containing the SQL query under the key 'sql_query'.

    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns an empty DataFrame in case of errors or no data.
    """
    
    sql_query = state["sql_query"]

    try:
        # Execute the SQL query using Supabase RPC
        response = supabase_client.rpc('execute_sql', {'query': sql_query}).execute()

        # Check if response contains data
        if response.data is None:
            print("No data returned from query.")
            return pd.DataFrame()

        # Create DataFrame from the data
        df = pd.DataFrame(response.data)

        # Convert date columns to datetime
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert numeric columns
        for col in df.columns:
            if '(000s)' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return {"data": df}

    except Exception as e:
        print(f"Error executing query: {str(e)}")
        print(f"Generated SQL query was: {sql_query}")
        return {"data": pd.DataFrame()}