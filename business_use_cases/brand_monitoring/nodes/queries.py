import json
from models import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from llm_client import client as llm
import re
from loguru import logger

def extract_queries(state: AgentState) -> AgentState:
    """Extract queries for the given themes to fetch relevant answers from answer engines."""
    logger.info("✨ Extracting queries for the given themes...")
    
    prompt = f"""
        Tasks:
        1. Analyse the themes extracted from the brand's webpage.
        2. For each theme, generate 3 relevant queries that users might ask conversational AI or search engines.
        3. Ensure the queries are specific to the brand's domain, industry, and location.
        4. Think of queries that would help users find information irrespective of the brand's name.
        4. Ensure the order of themes is from most important to least important.
        5. Return a JSON with the following format:

        # Output Format your response as JSON:
        {{
            "theme": ["<query 1>", "<query >","<query 3>"]    
        }}


        #INPUT
        URL: {state['url']}
        Domain of the brand: {state['domain']}
        Themes: {state['themes']}
        """    
    
    messages = [
        SystemMessage(content="You are a top content analyzer. Your job is to analyze the content of a brand's webpage, understand it's domain, and extract queries for SEO and conversational AI. Respond only with valid JSON."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages) 
        json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)        
        result = json.loads(json_str)        
        print(result)
        state["queries"] = result
        

        logger.info("✅ Extracted queries for the given themes.")
        logger.debug(f"Queries: {state['queries']}")
    except Exception as e:
        state["error"] = f"Content Analysis failed: {str(e)}"
        print(f"❌ Analysis error: {e}")
    
    return state