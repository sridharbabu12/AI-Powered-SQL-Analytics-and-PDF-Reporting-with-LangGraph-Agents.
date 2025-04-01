from pydantic import BaseModel, Field
from typing import List,Annotated,TypedDict,Dict,Any
import operator


class Analyst(BaseModel):
    name : str = Field(
        description = "name of the analyst for the analysis"
    )
    type : str = Field(
        description = "analysis type"
    )
    description : str = Field(
        description = "what this analysis will reveal"
    )
    required_columns : Annotated[List,operator.add] = Field(
        description = "add columns required for the analysis"
    )
    statistical_methods : Annotated[List,operator.add] = Field(
        description = "add methods to use for the analysis"
    )
    expected_insights : Annotated[List,operator.add] = Field(
        "Insights that we expect to find"
    )
    visualization_types : Annotated[List,operator.add] = Field(
        "visualization types used in this analysis"
    )
    
    
class Analysts(BaseModel):
    analysts : List[Analyst] = Field(
        description = "analysts for the analysis"
    )
    
    
class WorkerState(TypedDict):
    analyst : Analyst
    completed_analysts : Annotated[list[Analyst],operator.add]
    