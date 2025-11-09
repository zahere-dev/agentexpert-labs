import operator
from typing import Annotated, Optional, TypedDict


# State definition
class ProductPageState(TypedDict):
    """State that flows through the pipeline"""
    html_content: str
    url: str
    
    # Extracted data
    product_name: Optional[str]
    price: Optional[str]
    currency: Optional[str]
    description: Optional[str]
    images: Optional[list]
    sku: Optional[str]
    brand: Optional[str]
    availability: Optional[str]
    
    # Generated content
    json_ld_schema: Optional[str]
    optimized_title: Optional[str]
    optimized_description: Optional[str]
    conversational_keywords: Optional[list]
    meta_description: Optional[str]
    
    # Crawlability check
    crawlability_report: Optional[dict]
    robots_txt_content: Optional[str]
    
    # Validation
    is_valid: bool
    errors: Annotated[list, operator.add]
    
    # Output
    feed_data: Optional[dict]
    feed_xml: Optional[str]
    feed_csv: Optional[str]
    injection_complete: bool
