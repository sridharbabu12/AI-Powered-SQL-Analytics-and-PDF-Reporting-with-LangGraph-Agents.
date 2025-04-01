from projectai.LLMS.grogllm import model
from projectai.state.state import State
from projectai.state.analyst import Analysts
from langchain_core.messages import HumanMessage,SystemMessage

planner =model.with_structured_output(Analysts)


def aggregator(state:State):
    """
    combining the data from the dataretriever
    """
    
    # Check if 'data' key exists in the state
    if 'data' in state:
        combined = f"Here is the data collected from the database {state['data']} \n\n"
    else:
        combined = "No data available."
    
    #combined += f"and the data collected from the reddit {state['reddit_results']} "
    
    return {"combined_data":combined}


def orchestrator(state: State):
    """Orchestrator that generates analysts for the data analysis."""
    
    # Debugging: Check the combined data
    combined_data = state.get('combined_data', 'No combined data available.')
    print("Combined data:", combined_data)

    try:
        data_analysts = planner.invoke(
            [
                SystemMessage(content="Generate 2 analysts for the given data."),
                HumanMessage(content=f"Here is the data: {combined_data}")
            ]
        )
    except Exception as e:
        print("Error invoking planner:", e)
        return {"analysts": None}
    
    print("Data analysts:", data_analysts)
    
    # Check if analysts were generated successfully
    if hasattr(data_analysts, 'analysts'):
        return {"analysts": data_analysts.analysts}
    else:
        print("Failed to generate analysts. Response:", data_analysts)
        return {"analysts": None}