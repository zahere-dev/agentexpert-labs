import json
from openai import OpenAI
from models import AgentState
from llm_client import client as open_ai_client
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger


def openai_websearch(prompt:str, location: str = "US") -> str:

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
         web_search_options={
        "user_location": {
            "type": "approximate",
            "approximate": {
                "country": location,
                "city": "",
                "region": "",
            }
        },
    },
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    #print(completion.choices[0].message.content)
    return completion.choices[0].message


def get_brand_analysis_prompt(original_query: str,brand_name: str, answer_engine_response) -> str:
    """
    Generates a prompt to analyze how a brand fared in an answer engine response.

    Args:
        brand_name (str): The name of the brand to check in the response.
        answer_engine_response (str): The full text/content of the answer engine response.

    Returns:
        str: A formatted prompt string ready to use with an LLM.
    """
    prompt = f"""
        I want you to analyze the answer engine response to the query {original_query} and tell me how the brand '{brand_name}' fared. 
        If the brand was present in the response, provide the cited link. 
        If it was not the top answer, capture the responses and cited links of any competition above it. 
        If the brand is not present at all, provide the top 3 competitors with their responses and cited links.

        ## Answer Engine Response
        {answer_engine_response}

        ## Response format

        {{
        "brand_present": True,
        "brand_data": [{{
        "response": "",
        "citation": "",
        "rank": 1
        }}],
        "competition_data": [{{
        "name": "",
        "response": "",
        "citation": "",
        "rank": 1
        }}]
        }}
    """
    return prompt



def query_answer_engine(state) -> AgentState:
    """Fetch answers from answer engines for the given queries."""
    logger.info("✨ Fetching answers from answer engines for the given queries...")
    
    
    response_list = []
    questions_by_theme = state["queries"]

    for theme in questions_by_theme.keys():    
        theme_response = []
        queries = questions_by_theme[theme]
        for query in queries[:1]:        
            try:
                logger.info(f"🔍 Fetching answer for query: {query} under theme {theme}")            
                response = openai_websearch(query)
                
                prompt = get_brand_analysis_prompt(original_query=query,brand_name=state["brand_name"],answer_engine_response=response.content)
                
                messages = [
                        SystemMessage(content="You are a top brand analys. You will analyze the response from an answer engine (chatgpt, perplexity etc) and analyse the response as instructed. Respond only with valid JSON."),
                        HumanMessage(content=prompt)
                    ]
                
                response = open_ai_client.invoke(messages)
                import re
                json_str = re.search(r'\{.*\}', response.content, re.DOTALL).group(0)
                
                final_response = json.loads(json_str)        
                print(final_response)                
                query_response_dict = {"query":query, "response":final_response}
                theme_response.append(query_response_dict)
                logger.info(f"✅ Got response for {query}")
            except Exception as e:
                print(f"Error while getting respose for {query}: Error {e}")
                
        response_list.append({"theme":theme,"response":theme_response})
    state["query_results"] = response_list
    logger.info("✅ Fetched answers from answer engines for the given queries.")
    logger.debug(f"Query Results: {state['query_results']}")
    return state
        