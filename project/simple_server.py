from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List
import os
from supabase import create_client, Client
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Create a named MCP server
mcp = FastMCP("Supabase Query Server")

# Initialize clients
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL", ""),
    os.environ.get("SUPABASE_KEY", "")
)

claude_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
)

# Define schema information resource
@mcp.resource("schema://tables")
async def get_schema() -> Dict[str, List[Dict[str, Any]]]:
    """Get database schema information"""
    try:
        # Query to get table names
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        tables_response = supabase.rpc('execute_sql', {'query': tables_query}).execute()
        
        schema_info = {}
        for table in [row['table_name'] for row in tables_response.data]:
            # Query to get column information
            columns_query = f"""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = '{table}'
            """
            columns_response = supabase.rpc('execute_sql', {'query': columns_query}).execute()
            schema_info[table] = columns_response.data
        
        return {
            "success": True,
            "schema": schema_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Define natural language to SQL conversion tool
@mcp.tool()
async def convert_to_sql(natural_query: str) -> Dict[str, Any]:
    """
    Convert natural language query to SQL using Claude
    
    Args:
        natural_query: Natural language description of the query
    
    Returns:
        Dictionary containing the generated SQL query
    """
    try:
        # Get schema first
        schema_info = await get_schema()
        if not schema_info["success"]:
            return {
                "success": False,
                "error": "Failed to fetch schema information"
            }

        # Create schema description for Claude
        schema_description = "Database Schema:\n"
        for table, columns in schema_info["schema"].items():
            schema_description += f"\nTable: {table}\n"
            for col in columns:
                col_name = col['column_name']
                col_type = col['data_type']
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                schema_description += f"  - {col_name} ({col_type}, {nullable})\n"

        # Send request to Claude
        message = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            system="""
            You are an expert PostgreSQL query generator. Generate valid PostgreSQL queries for a Supabase database.
            
            Important rules:
            1. ONLY use columns that exist in the schema
            2. Use exact column names as shown in the schema (case-sensitive)
            3. Always enclose table and column names in double quotes when they:
               - Contain spaces
               - Contain special characters
               - Use mixed case
               - Are PostgreSQL keywords
            4. Do NOT include semicolons at the end of queries
            5. Return ONLY the SQL query without any explanations
            """,
            messages=[{
                "role": "user",
                "content": f"""
                {schema_description}
                
                Convert this natural language query to SQL:
                {natural_query}
                """
            }]
        )

        sql_query = message.content[0].text.strip()
        
        return {
            "success": True,
            "sql_query": sql_query,
            "natural_query": natural_query
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Define a query execution tool
@mcp.tool()
async def execute_query(natural_query: str) -> Dict[str, Any]:
    """
    Convert and execute a natural language query
    
    Args:
        natural_query: Natural language description of the query
    
    Returns:
        Dictionary containing query results
    """
    try:
        # First convert to SQL
        conversion_result = await convert_to_sql(natural_query)
        if not conversion_result["success"]:
            return conversion_result
        
        # Execute the SQL query
        sql_query = conversion_result["sql_query"]
        response = supabase.rpc('execute_sql', {'query': sql_query}).execute()
        
        return {
            "success": True,
            "natural_query": natural_query,
            "sql_query": sql_query,
            "data": response.data,
            "count": len(response.data) if response.data else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Define a table-specific query resource
@mcp.resource("table://{table_name}/limit/{limit}")
async def get_table_data(table_name: str, limit: str = "10") -> Dict[str, Any]:
    """
    Get data from a specific table
    
    Args:
        table_name: Name of the table to query
        limit: Maximum number of rows to return (as string)
    """
    try:
        # Convert limit to integer
        limit_int = int(limit)
        query = f'SELECT * FROM "{table_name}" LIMIT {limit_int}'
        response = supabase.rpc('execute_sql', {'query': query}).execute()
        return {
            "success": True,
            "table": table_name,
            "data": response.data,
            "count": len(response.data) if response.data else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Update the prompt template
@mcp.prompt()
def database_help(query: str) -> str:
    """Help template for database queries"""
    return f"""
    User needs help with: {query}
    
    Available tools:
    - convert_to_sql: Convert natural language to SQL query
    - execute_query: Convert and execute natural language query
    - query_database: Execute raw SQL query
    
    Available resources:
    - schema://tables: Get database schema information
    - table://{{table_name}}: Get data from a specific table
    
    Example usage:
    1. Get schema information:
       - Use schema://tables to understand database structure
    
    2. Convert natural language to SQL:
       - Use convert_to_sql tool with your question
       Example: "Show me all users who joined last month"
    
    3. Execute natural language query:
       - Use execute_query tool to convert and run your query
       Example: "List the top 5 products by sales"
    
    4. Query specific table:
       - Use table://users to get user data
    
    5. Execute raw SQL:
       - Use query_database tool with SQL query
    """

if __name__ == "__main__":
    # To run the server:
    # Option 1: mcp install simple_server.py
    # Option 2: mcp dev simple_server.py
    pass
