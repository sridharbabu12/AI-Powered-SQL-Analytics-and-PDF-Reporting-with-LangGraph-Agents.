from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from projectai.state.state import State
from projectai.nodes.schema import get_schema_info
from projectai.nodes.sqlquerygenerator import generate_sql_from_query
from projectai.nodes.dataretriever import query_database
from projectai.nodes.orchestrator import orchestrator,aggregator
from projectai.tools.reddit import get_reddit_text
from projectai.nodes.llm_workers import llm_workers,assign_workers
from projectai.nodes.pdfsynthesizer import pdfsynthesizer
from IPython.display import Markdown
import asyncio  # Import asyncio to manage the event loop

class GraphBuilder:
    """Builds and manages the automation workflow graph."""
    
    def __init__(self):
        self.parallel_builder = StateGraph(State)
    
    async def graph(self) -> None:
        """
        Builds automation graph using langgraph.
        """
        # Add nodes with their corresponding functions
        
        self.parallel_builder.add_node("table_schema", get_schema_info )
        self.parallel_builder.add_node("generate_sql_query", generate_sql_from_query)
        self.parallel_builder.add_node("dataretriever", query_database)
        self.parallel_builder.add_node("aggregator", aggregator)
        self.parallel_builder.add_node("orchestrator", orchestrator)
        #self.parallel_builder.add_node("reddit_search", get_reddit_text)
        self.parallel_builder.add_node("llm_workers",llm_workers)
        self.parallel_builder.add_node("pdfsynthesizer",pdfsynthesizer)
        
        # Add edges to connect nodes
        self.parallel_builder.add_edge(START, "table_schema")
        #self.parallel_builder.add_edge(START, "reddit_search")
        self.parallel_builder.add_edge("table_schema", "generate_sql_query")
        self.parallel_builder.add_edge("generate_sql_query", "dataretriever")
        self.parallel_builder.add_edge("dataretriever", "aggregator")
        #self.parallel_builder.add_edge("reddit_search", "aggregator")
        self.parallel_builder.add_edge("aggregator","orchestrator")
        self.parallel_builder.add_conditional_edges("orchestrator",assign_workers,["llm_workers"])
        self.parallel_builder.add_edge("llm_workers","pdfsynthesizer")
        self.parallel_builder.add_edge("pdfsynthesizer", END)
        
        parallel_workflow = self.parallel_builder.compile()
        
        display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
        
        await parallel_workflow.ainvoke({'user_query': "What insights can be drawn from the National AA (000s) data in terms of audience engagement and growth over time?"})
    

graph_builder = GraphBuilder()

# Define an asynchronous main function
async def main():
    await graph_builder.graph()  # Await the graph method

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main()) 
    # Ensure that State["final_analysis"] is accessed correctly
    final_analysis_text = str(State["completed_analysts"]) if hasattr(State, "completed_analysts") else "No analysis available."  # Default message if key doesn't exist
    Markdown(final_analysis_text) # Use display to show the Markdown