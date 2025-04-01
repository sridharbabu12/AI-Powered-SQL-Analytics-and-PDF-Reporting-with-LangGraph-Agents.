from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

def generate_hypotheses(prompt, num_hypotheses=3):
    """
    Generate multiple hypotheses (dish ideas) using the LLM.
    """
    messages = [
        SystemMessage(content="You are a culinary expert specializing in South Indian cuisine. You experiment with ingredients to create new flavors."),
        HumanMessage(content=f"Propose {num_hypotheses} unique South Indian dishes using traditional spices but with an innovative twist.")
    ]
    response = llm.invoke(messages)
    hypotheses = response.content.strip().split("\n")
    return [h.strip() for h in hypotheses if h.strip()]

def evaluate_hypothesis(hypothesis):
    """
    Evaluate a hypothesis (dish idea) using the LLM.
    """
    messages = [
        SystemMessage(content="You are a food scientist who analyzes the balance of flavors in dishes."),
        HumanMessage(content=f"Analyze the following dish idea and rate its creativity, flavor balance, and feasibility on a scale of 1 to 10: {hypothesis}")
    ]
    evaluation = llm.invoke(messages)
    return evaluation.content

def refine_hypothesis(hypothesis, feedback):
    """
    Refine a hypothesis based on feedback.
    """
    messages = [
        SystemMessage(content="You are a culinary expert who refines dishes based on feedback."),
        HumanMessage(content=f"Refine the following dish idea based on this feedback: {feedback}. Dish: {hypothesis}")
    ]
    refined_dish = llm.invoke(messages)
    return refined_dish.content

def tree_of_thought(prompt, num_hypotheses=2, max_iterations=1):
    """
    Use the Tree of Thought approach to create and refine a South Indian dish.
    """
    # Step 1: Generate multiple hypotheses
    hypotheses = generate_hypotheses(prompt, num_hypotheses)
    print("Generated Hypotheses:")
    for i, hypothesis in enumerate(hypotheses):
        print("-------------------------------------------------------------")
        print(f"{i + 1}. {hypothesis}")
    
    # Step 2: Evaluate hypotheses
    evaluations = []
    for hypothesis in hypotheses:
        evaluation = evaluate_hypothesis(hypothesis)
        evaluations.append((hypothesis, evaluation))
        print("-------------------------------------------------------------")
        print(f"Evaluation for '{hypothesis}': {evaluation}")
    
    # Step 3: Select the best hypothesis
    best_hypothesis = max(evaluations, key=lambda x: len(x[1]))[0] 
    print("-------------------------------------------------------------")# Select based on evaluation length (simplified)
    print(f"\nSelected Best Hypothesis: {best_hypothesis}")
    
    # Step 4: Refine the best hypothesis iteratively
    for iteration in range(max_iterations):
        print("-------------------------------------------------------------")
        print(f"\nIteration {iteration + 1}: Refining the dish...")
        feedback = evaluate_hypothesis(best_hypothesis)
        best_hypothesis = refine_hypothesis(best_hypothesis, feedback)
        print("-------------------------------------------------------------")
        print(f"Refined Dish: {best_hypothesis}")
    
    # Step 5: Finalize the recipe
    final_prompt = f"Provide a step-by-step recipe for the following dish: {best_hypothesis}"
    messages = [
        SystemMessage(content="You are a culinary expert who provides detailed recipes."),
        HumanMessage(content=final_prompt)
    ]
    final_recipe = llm.invoke(messages)
    print("-------------------------------------------------------------")
    print(f"\nFinal Recipe:\n{final_recipe.content}")

# Define the user prompt
user_prompt = "Create a unique South Indian dish using traditional spices but with an innovative twist."

# Run the Tree of Thought process
tree_of_thought(user_prompt)