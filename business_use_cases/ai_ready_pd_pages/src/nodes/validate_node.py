import json
from models import ProductPageState


def validate(state: ProductPageState) -> ProductPageState:
    """Agent 4: Validate schema and content completeness"""
    print("🔍 Validating...")
    
    errors = []
    
    # Check required fields
    if not state.get("product_name"):
        errors.append("Missing product name")
    if not state.get("price"):
        errors.append("Missing price")
    if not state.get("json_ld_schema"):
        errors.append("Missing JSON-LD schema")
    if not state.get("optimized_description"):
        errors.append("Missing optimized description")
    
    # Validate JSON-LD structure
    if state.get("json_ld_schema"):
        try:
            schema = json.loads(state["json_ld_schema"])
            if "@context" not in schema or "@type" not in schema:
                errors.append("Invalid JSON-LD structure")
        except:
            errors.append("JSON-LD is not valid JSON")
    
    state["errors"] = errors
    state["is_valid"] = len(errors) == 0
    
    if state["is_valid"]:
        print("✅ Validation passed")
    else:
        print(f"❌ Validation failed: {errors}")
    
    return state