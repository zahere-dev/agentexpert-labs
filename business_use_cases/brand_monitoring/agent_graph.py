from datetime import datetime
import json
from langgraph.graph import StateGraph, END
from loguru import logger

from dotenv import load_dotenv
load_dotenv()


from models import AgentState
from nodes.scraper import scraper_node
from nodes.themes import extract_themes
from nodes.queries import extract_queries
from nodes.query_answer_engine import query_answer_engine




def create_pipeline():
    workflow = StateGraph(AgentState)
    workflow.add_node("scrape", scraper_node)
    workflow.add_node("themes", extract_themes)
    workflow.add_node("queries", extract_queries)
    workflow.add_node("answer_engine", query_answer_engine)

    
    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "themes")
    workflow.add_edge("themes", "queries")
    workflow.add_edge("queries", "answer_engine")
    workflow.add_edge("answer_engine", END)

    return workflow.compile()


brand_monitor_graph = create_pipeline()
def run_brand_monitor_graph(url: str) -> dict:    
    """Run the brand monitoring agent graph."""
        
    logger.info(f"🚀 Running brand monitoring agent graph for URL: {url}")
    
    initial_state: AgentState = {
        "url": url,
        "html_content": None,
        "domain": None,
        "error": None,
        "themes": None,
        "queries": None,
        "query_results": None,
        "analysis_report": None,
        "location": None,
        "brand_name": None
    }
    
    final_state = brand_monitor_graph.invoke(initial_state)    
    
    # write to json file
    with open(f'brand_monitoring_result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json', 'w') as f:
        json.dump(final_state["query_results"], f)
    
    logger.info("✅ Brand monitoring agent graph completed.")    
    return final_state


run_brand_monitor_graph("https://www.michelinman.com/")