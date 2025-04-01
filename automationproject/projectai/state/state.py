from typing import TypedDict, Dict, List, Any, Annotated
import pandas as pd
import operator
from projectai.state.analyst import Analyst

class State(TypedDict):
    """
    State representation for the chatbot workflow
    
    Attributes:
    - user_query: The user's query for information retrieval
    - schema_info: Schema information for the database
    - sql_query: The generated SQL query
    - data: Retrieved data from database sources
    - combined_data: Combined results from various sources
    - analysts: List of analysts involved in the process
    - completed_analysts: List of completed analysts
    - final_analysis: Final analysis result
    """
    
    user_query: str
    schema_info: Dict[str, List[Dict[str, Any]]]
    sql_query: str
    data: pd.DataFrame
    #reddit_results: List[Dict[str, Any]]
    combined_data: List[Dict[str, Any]]
    analysts: List[Analyst]
    completed_analysts: Annotated[List[Analyst], operator.add]
    final_analysis: str
    

    