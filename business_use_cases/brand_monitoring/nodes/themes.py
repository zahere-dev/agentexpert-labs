import json
from models import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from llm_client import client as llm
from loguru import logger

def extract_themes(state: AgentState) -> AgentState:
    """Extract themes for SEO and conversational AI from the content of a given page."""
    logger.info("✨ Extract themes for SEO and conversational AI...")
    
    prompt = f"""
Tasks:
1. Analyze the content of a brand's webpage, understand it's domain, industry, location 
2. Extract key themes for SEO and conversational AI
3. Think of all the themes that users in that domain may be interested in. Do take into account the industry and location.
4. Ensure there are 3 themes and the order of themes is from most important to least important.
5. Return a JSON with the following format:

# Output Format your response as JSON:
{{
    "domain": "<Description of the webpage's domain and industry>",
    "brand_name": "<Name of the brand/business>",
    "location": "<Location code of the brand/business Ex: US, IN, UK>",
    "themes": ["...", "...", ...]    
}}


#INPUT
URL: {state['url']}
Content: {state['html_content']}
"""    
    
    messages = [
        SystemMessage(content="You are a top content analyzer. Your job is to analyze the content of a brand's webpage, understand it's domain, and extract themes for SEO and conversational AI. Respond only with valid JSON."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages) 
        
        import re
        json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)
        
        result = json.loads(json_str)        
        print(result)
        state["domain"] = result["domain"]
        state["location"] = result["location"]
        state["brand_name"] = result["brand_name"]
        state["themes"] = result["themes"]       

        logger.info("✅ Content Analyzed and themes extracted.")
        logger.debug(f"Domain: {state['domain']}, Location: {state['location']}, Brand Name: {state['brand_name']}, Themes: {state['themes']}")
    except Exception as e:
        state["error"] = f"Content Analysis failed: {str(e)}"
        print(f"❌ Analysis error: {e}")
    
    return state