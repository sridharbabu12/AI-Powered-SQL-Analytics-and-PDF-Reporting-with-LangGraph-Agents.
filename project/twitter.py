from agno.agent import Agent 
from agno.tools.x import XTools
from dotenv import load_dotenv

load_dotenv()


x_tools=XTools()

agent=Agent(
    instructions=[
        "You are authorized to interact with X (Twitter) as the authenticated user.",
        "Use your tools to search, retrieve, and analyze content from X.",
        "For any search operation, fetch a maximum of **10 tweets only** to avoid hitting rate limits.",
        "When asked about a topic or show, perform the following:",
        "   - Search for recent tweets  using relevant keywords or hashtags.",
        "   - Summarize the sentiment (positive, neutral, negative) of the fetched tweets.",
        "   - Report engagement highlights (likes, retweets, replies) if available.",
        "   - Generate sample content or a summary if requested.",
        "Always respect X's usage policies, developer terms, and rate limits."
    ],
    tools=[x_tools],
    show_tool_calls=True
)

agent.print_response("Search tweets about the BET show sistas or #sistas or #BET and summarize the audience sentiment and engagement.",markdown=True)


