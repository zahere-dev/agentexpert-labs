import json
from models import ProductPageState
from langchain_core.messages import HumanMessage, SystemMessage
from llm_client import client as llm

def optimize_content(state: ProductPageState) -> ProductPageState:
    """Agent 3: Optimize content for natural language and conversational AI"""
    print("✨ Optimizing content for natural language...")
    
    prompt = f"""You are an e-commerce SEO expert optimizing product content for AI answer engines like ChatGPT.

Product: {state['product_name']}
Current Description: {state['description']}
Price: {state['price']}

Tasks:
1. Create a conversational, natural-language product title (not robotic)
2. Rewrite the description in natural language that answers "what is this product best for?"
3. Generate 5 conversational keyword phrases (e.g., "perfect for", "ideal for", "best for")
4. Create a compelling meta description (150-160 chars)

Format your response as JSON:
{{
    "optimized_title": "...",
    "optimized_description": "...",
    "conversational_keywords": ["...", "...", ...],
    "meta_description": "..."
}}
"""
    
    messages = [
        SystemMessage(content="You are an SEO expert. Respond only with valid JSON."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        result = json.loads(response.content)
        state["optimized_title"] = result["optimized_title"]
        state["optimized_description"] = result["optimized_description"]
        state["conversational_keywords"] = result["conversational_keywords"]
        state["meta_description"] = result["meta_description"]
        print("✅ Content optimized")
    except Exception as e:
        state["errors"] = [f"Content optimization failed: {str(e)}"]
        print(f"❌ Optimization error: {e}")
    
    return state