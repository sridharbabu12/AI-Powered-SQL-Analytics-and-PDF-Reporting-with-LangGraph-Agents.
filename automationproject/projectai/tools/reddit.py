from typing import List
import praw
from projectai.LLMS.grogllm import model
from projectai.state.state import State

def get_reddit_text(state:State):
    """
    Search Reddit and return only the text content from posts.
    
    Args:
        query: Search query string
        subreddit: Subreddit to search (defaults to "all")
        limit: Maximum number of results to return
        
    Returns:
        List of text content from posts
    """
    # Initialize Reddit client with your credentials
    reddit = praw.Reddit(
        client_id = "kN0XLzaIG7i3qJ-5p5q1ag",
        client_secret = "hDoJe2SIhMnsINEjshv5ffcGHLPGog",
        user_agent = "python:aiproject:v1.0 (by /u/Acceptable_Rub_5830"
    )
    
    # Search the specified subreddit
    subreddit = reddit.subreddit("Sistas")
    search_results = subreddit.search("sistas", 5)
    
    # Extract only text content from posts
    text_contents = []
    for post in search_results:
        # Get post text (selftext) if it exists
        if post.selftext:
            text_contents.append(post.selftext)
        # If no selftext, use title
        elif post.title:
            text_contents.append(post.title)
            
    max_length = 1000  # Set a maximum length for each text entry
    limited_text_contents = [text[:max_length] for text in text_contents]
            
    result=model.invoke(f"by analyzing the text contents {limited_text_contents} make a report on what are they taking about?")
    return {"reddit_results":result}
