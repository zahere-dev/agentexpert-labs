import json
from models import ProductPageState
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from llm_client import client as llm


def extract_content(state: ProductPageState) -> ProductPageState:
    """Agent 1: Extract product data from HTML using LLM for intelligent extraction"""
    print("🔍 Extracting content from page...")
    
    soup = BeautifulSoup(state["html_content"], 'html.parser')
    
    # Remove script, style, and other non-content elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Get clean text content
    page_text = soup.get_text(separator='\n', strip=True)
    
    # Limit text to avoid token limits (keep first 3000 chars which is ~750 tokens)
    page_text_limited = page_text[:3000]
    
    # Extract images with basic heuristics
    images = []
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src')
        if src:
            # Convert relative URLs to absolute
            if src.startswith('http'):
                images.append(src)
            elif src.startswith('//'):
                images.append('https:' + src)
            elif src.startswith('/'):
                from urllib.parse import urlparse
                parsed = urlparse(state["url"])
                images.append(f"{parsed.scheme}://{parsed.netloc}{src}")
            else:
                # Handle relative paths like ../../
                from urllib.parse import urljoin
                images.append(urljoin(state["url"], src))
    
    # Use LLM to extract structured product information
    extraction_prompt = f"""You are a product data extraction expert. Extract product information from the following webpage content.

URL: {state["url"]}

Page Content:
{page_text_limited}

Extract the following information and respond ONLY with valid JSON (no markdown, no explanations):
{{
    "product_name": "the main product title or name",
    "price": "the price with currency symbol (e.g., $29.99, £19.99)",
    "currency": "3-letter currency code (USD, GBP, EUR, etc.)",
    "description": "product description (2-3 sentences max)",
    "brand": "brand or manufacturer name",
    "sku": "SKU, UPC, product ID, or model number if available",
    "availability": "InStock or OutOfStock based on availability indicators"
}}

Rules:
- If any field is not found, use null
- For currency, extract from price symbol: $ = USD, £ = GBP, € = EUR
- For availability, look for phrases like "in stock", "available", "out of stock", "sold out"
- Keep descriptions concise and factual
"""
    
    messages = [
        SystemMessage(content="You extract structured product data. Respond only with valid JSON."),
        HumanMessage(content=extraction_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        # Clean response in case LLM adds markdown
        response_text = response.content.strip()
        if response_text.startswith('```'):
            # Remove markdown code blocks
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        extracted = json.loads(response_text)
        
        state["product_name"] = extracted.get("product_name")
        state["price"] = extracted.get("price")
        state["currency"] = extracted.get("currency") or "USD"
        state["description"] = extracted.get("description")
        state["brand"] = extracted.get("brand") or "Unknown Brand"
        state["sku"] = extracted.get("sku") or f"SKU-{abs(hash(state['product_name']))}"
        state["availability"] = extracted.get("availability") or "InStock"
        state["images"] = images[:5]  # Keep up to 5 images
        
        print(f"✅ Extracted: {state['product_name']}")
        print(f"   Price: {state['price']} ({state['currency']})")
        print(f"   Brand: {state['brand']}")
        print(f"   Availability: {state['availability']}")
        print(f"   Images found: {len(state['images'])}")
        
    except Exception as e:
        print(f"❌ LLM extraction failed: {e}")
        # Fallback to basic extraction
        state["product_name"] = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Unknown Product"
        state["price"] = "0.00"
        state["currency"] = "USD"
        state["description"] = page_text_limited[:200]
        state["brand"] = "Unknown Brand"
        state["sku"] = f"SKU-{abs(hash(state['product_name']))}"
        state["availability"] = "InStock"
        state["images"] = images[:5]
        state["errors"] = [f"LLM extraction failed: {str(e)}"]
    
    return state