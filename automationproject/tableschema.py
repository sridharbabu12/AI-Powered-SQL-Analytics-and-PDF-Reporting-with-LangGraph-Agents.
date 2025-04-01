from dotenv import load_dotenv
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.tools.reddit_search.tool import RedditSearchSchema
#from projectai.LLMS.grogllm import model
load_dotenv()

client_id = "kN0XLzaIG7i3qJ-5p5q1ag"
client_secret = "hDoJe2SIhMnsINEjshv5ffcGHLPGog"
user_agent = "python:aiproject:v1.0 (by /u/Acceptable_Rub_5830"
def reddit_search():
    """
    it searches the reddit about the details of the given query 
    """
    reddit_wrapper = RedditSearchAPIWrapper(
        reddit_client_id=client_id,
        reddit_client_secret=client_secret,
        reddit_user_agent=user_agent,
    )
    search = RedditSearchRun(api_wrapper=reddit_wrapper) 
    search_params = RedditSearchSchema(
    query="*", sort="comments", time_filter="month", subreddit="Sistas", limit="5")
    result = search.run(tool_input=search_params.model_dump())
    
    return print(result)
reddit_search()