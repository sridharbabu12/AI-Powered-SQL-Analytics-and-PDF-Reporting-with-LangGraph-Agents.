from projectai.state.state import State
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import matplotlib.pyplot as plt
import seaborn as sns

def pdfsynthesizer(state:State):
    """
    synthesize full analysis from analysts
    
    """
 

    completed_analysts = state["completed_analysts"]
    
    completed_report_analysis = "\n\n-----\n\n".join(completed_analysts)
    
    return {"final_analysis": completed_report_analysis}