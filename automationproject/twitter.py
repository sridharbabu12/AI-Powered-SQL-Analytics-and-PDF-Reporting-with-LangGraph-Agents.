from agno.agent import Agent 
from agno.tools.x import XTools
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

def create_twitter_search_agent(search_instructions: Optional[str] = None) -> Agent:
    """
    Creates an agent configured for Twitter/X searches with custom instructions.
    
    Args:
        search_instructions: Optional custom instructions for the agent
    
    Returns:
        Agent: Configured Twitter search agent
    """
    x_tool = XTools()
    
    default_instructions = """
    Use your tools to search for content on X (Twitter).
    When searching:
    1. Perform searches with the provided query
    2. Analyze and summarize relevant tweets
    3. Respect X's usage policies and rate limits
    4. Provide informative responses based on search results
    """
    
    agent = Agent(
        instructions=[search_instructions or default_instructions],
        tools=[x_tool],
        show_tool_calls=True,
    )
    return agent

def search_twitter(query: str, custom_instructions: Optional[str] = None) -> str:
    """
    Performs a Twitter search for the given query.
    
    Args:
        query: Search query string
        custom_instructions: Optional custom instructions for the search
    
    Returns:
        str: Search results and analysis
    """
    agent = create_twitter_search_agent(custom_instructions)
    return agent.print_response(f"Search and analyze tweets about: {query}", markdown=True)

# Example usage
if __name__ == "__main__":
    # Example 1: Basic search
    results = search_twitter("BET sistas show")
    print(results)
