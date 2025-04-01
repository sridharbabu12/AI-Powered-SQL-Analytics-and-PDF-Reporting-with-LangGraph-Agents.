from projectai.state.state import State
from projectai.LLMS.grogllm import model
from projectai.tools.reddit import reddit_tool
#from projectai.tools.twitter import x_tool
class Agent():
    """
    Invokes the agent model to generate a response based on the current state.  
    Given the user input, the agent will decide whether to refine the query,  
    retrieve relevant data using external search tools, or aggregate and respond.

    Args:
        state (dict): The current state containing the user query and message history.

    Returns:
        dict: The updated state with the agent response appended to messages.
    """
    
    
    def __init__(self,model):
        self.llm=model
        
    def process(self,state:State) -> dict:
        """
        """
        return {"messages" : self.llm.invoke(state['messages'])}

tools=[reddit_tool]
model.bind_tools(tools)
agent=Agent(model)
# Create a sample state (assuming it follows a dict structure)
state = State({'messages': "tell me about the shows of BET"})

# Run the agent
response = agent.process(state)
print(response)

    
    
    for viz in state["completed_analysts"].visualization_types:
        plt.figure(figsize=(10, 6))
        try:
                # Clean up the code string
            code = viz["code"].strip()
            if code.startswith('```python'):
                code = code.split('```python')[1]
            if code.endswith('```'):
                code = code[:-3]
                    
                    # Execute the visualization code
                exec(code, {
                        'df': df, 
                        'plt': plt, 
                        'sns': sns, 
                        'pd': pd,
                        'np': np
                    })
                    
                    # Capture the current figure
                fig = plt.gcf()
                figures.append(fig)
                    
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            plt.close()
            continue

        return {
                "title": analysis_plan.get("title", "Data Analysis"),
                "figures": figures,
                "insights": analysis_plan.get("insights", []),
                "analysis_plan": analysis_plan,
                "data": df
            }
            
    
    
    
    
    
    
    
    
    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = state[""]
    slide.placeholders[1].text = f"Data Analysis Report\nGenerated on {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    
    # Key Insights Slide
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    slide.shapes.title.text = "Key Insights"
    
    tf = slide.placeholders[1].text_frame
    insights = analysis_results.get("insights", [])
    insights = [insights] if isinstance(insights, str) else insights
    
    for insight in insights[:5]:  # Limit to 5 insights
        p = tf.add_paragraph()
        p.text = insight
        p.level = 0
    
    # Visualization Slides
    for i, fig in enumerate(analysis_results.get("figures", [])):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        slide_layout = prs.slide_layouts[5]  # Title and Content layout
        slide = prs.slides.add_slide(slide_layout)
        
        title = analysis_results.get("analysis_plan", {}).get("visualizations", [{}])[i].get("description", f"Visualization {i+1}")
        slide.shapes.title.text = title
        
        slide.shapes.add_picture(buf, Inches(0.5), Inches(1.5), width=Inches(6))
        
        analysis_box = slide.shapes.add_textbox(Inches(7), Inches(1.5), Inches(3), Inches(5))
        tf = analysis_box.text_frame
        tf.word_wrap = True
        
        p = tf.add_paragraph()
        p.text = "Analysis"
        p.font.bold = True
        p.font.size = Pt(14)
        
        observations = analysis_results.get("analysis_plan", {}).get("visualizations", [{}])[i].get("analysis", [])
        observations = [observations] if isinstance(observations, str) else observations
        
        for obs in observations:
            p = tf.add_paragraph()
            p.text = f"• {obs}"
            p.font.size = Pt(11)
        
        plt.close(fig)
    
    # Data Summary Slide
    df = analysis_results.get("data")
    if df is not None and not df.empty and len(df.columns) <= 10:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = "Data Summary"
        
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        tf = txBox.text_frame
        tf.word_wrap = True
        
        p = tf.add_paragraph()
        p.text = " | ".join(map(str, df.columns))
        p.font.bold = True
        
        for _, row in df.head(10).iterrows():
            p = tf.add_paragraph()
            p.text = " | ".join(map(str, row))
    
    # Final Slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    txBox = slide.shapes.add_textbox(Inches(2.5), Inches(3), Inches(5), Inches(1))
    tf = txBox.text_frame
    
    p = tf.add_paragraph()
    p.text = "Thank You"
    p.alignment = 1  # Center
    p.font.size = Pt(40)
    
    return prs