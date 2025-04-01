from projectai.state.analyst import WorkerState
from projectai.LLMS.grogllm import model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from projectai.state.state import State


def llm_workers(state: WorkerState):
    """
    workers writes an analysis of the data 
    """
    # Check if 'data' exists in the state
    
    analyst_info = f"""
    Here is the name for analyst: {state['analyst'].name}, type: {state['analyst'].type}, description: {state['analyst'].description}, 
    required columns: {state['analyst'].required_columns}, 
    statistical methods: {state['analyst'].statistical_methods}, 
    expected insights: {state['analyst'].expected_insights}, 
    visualization types: {state['analyst'].visualization_types} and here is the data for analysis {State['combined_data']}
    """
    
    print("Analyst Info:", analyst_info)  # Log the analyst information

    analyst = model.invoke(
        [
            SystemMessage(
                content="""
                You are an expert data visualization developer. Generate Python code using matplotlib and seaborn
                to create insightful visualizations based on the data and user query do not use sample dataset only use the original dataset.dont assume anything extra just do the task by using the given data.
                
                The visualizations should:
                1. Use clear, professional styling with proper labels and titles
                2. Include appropriate color schemes and legends
                3. Handle data types correctly (dates, numbers, categories)
                4. Use the most suitable chart types for the analysis
                5. Show meaningful insights about the data
                
                For each visualization, include 2-3 specific observations about:
                - Trends or patterns shown in the data
                - Notable outliers or anomalies
                - Relationships between variables
                - Business implications of the findings
                """
            ),
            HumanMessage(content=analyst_info)
        ]
    )
    
    print("Model Response:", analyst)  # Log the model response
    
    # Update the state with the completed analyst
    if 'completed_analysts' in state:
        state['completed_analysts'].append(analyst.content)
    else:
        state['completed_analysts'] = [analyst.content]
    
    return {"completed_analysts": state['completed_analysts']}


def assign_workers(state: State):
    """
    assign a worker to each analyst in the plan
    """
    # Ensure that state1 is passed correctly
    return [Send("llm_workers", {"analyst": a}) for a in state['analysts']]