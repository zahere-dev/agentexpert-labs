import json
from models import ProductPageState
from langchain_core.messages import HumanMessage, SystemMessage

def generate_schema(state: ProductPageState) -> ProductPageState:
    """Agent 2: Generate JSON-LD schema"""
    print("📋 Generating JSON-LD schema...")
    
    schema = {
        "@context": "https://schema.org/",
        "@type": "Product",
        "name": state["product_name"],
        "description": state["description"],
        "sku": state["sku"],
        "brand": {
            "@type": "Brand",
            "name": state["brand"]
        },
        "image": state["images"],
        "offers": {
            "@type": "Offer",
            "url": state["url"],
            "priceCurrency": state["currency"],
            "price": state["price"].replace('$', '').strip() if state["price"] else "0.00",
            "itemCondition": "https://schema.org/NewCondition",
            "availability": f"https://schema.org/{state['availability']}"
        }
    }
    
    state["json_ld_schema"] = json.dumps(schema, indent=2)
    print("✅ Schema generated")
    return state
