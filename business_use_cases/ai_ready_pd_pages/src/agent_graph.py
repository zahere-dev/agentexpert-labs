import json
from langgraph.graph import StateGraph, END
import requests

from dotenv import load_dotenv
load_dotenv()

from models import ProductPageState
from nodes.extract_content_node import extract_content
from nodes.generate_schema_node import generate_schema
from nodes.inject_to_page_node import inject_to_page
from nodes.monitor_and_log_node import monitor_and_log
from nodes.optimize_content_node import optimize_content
from nodes.check_crawlability_node import check_crawlability
from nodes.validate_node import validate

def should_inject(state: ProductPageState) -> str:
    """Router: Decide whether to inject or skip"""
    if state["is_valid"]:
        return "inject"
    else:
        return "skip"

def log_error(state: ProductPageState) -> ProductPageState:
    """Log errors and skip injection"""
    print(f"⚠️  Skipping injection due to errors: {state['errors']}")
    state["injection_complete"] = False
    return state



def create_pipeline():
    """Create the LangGraph pipeline"""
    workflow = StateGraph(ProductPageState)
    
    # Add nodes
    workflow.add_node("extract", extract_content)
    workflow.add_node("check_crawlability", check_crawlability)
    workflow.add_node("generate_schema", generate_schema)
    workflow.add_node("optimize", optimize_content)
    workflow.add_node("validate", validate)
    workflow.add_node("inject", inject_to_page)
    workflow.add_node("log_error", log_error)
    workflow.add_node("monitor", monitor_and_log)
    
    # Define edges
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "check_crawlability")
    workflow.add_edge("check_crawlability", "generate_schema")
    workflow.add_edge("generate_schema", "optimize")
    workflow.add_edge("optimize", "validate")
    
    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validate",
        should_inject,
        {
            "inject": "inject",
            "skip": "log_error"
        }
    )
    
    workflow.add_edge("inject", "monitor")
    workflow.add_edge("log_error", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()




def process_single_product(url: str, app, output_dir: str = "output") -> dict:
    """Process a single product URL"""
    import requests
    from datetime import datetime
    import os
    from urllib.parse import urlparse
    
    print(f"\n{'='*70}")
    print(f"Processing: {url}")
    print('='*70)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html_content = response.text
        
        # Run pipeline
        initial_state = {
            "html_content": html_content,
            "url": url,
            "is_valid": False,
            "errors": [],
            "injection_complete": False
        }
        
        result = app.invoke(initial_state)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate safe filename from URL
        parsed = urlparse(url)
        safe_name = parsed.path.strip('/').replace('/', '_')[:50]
        if not safe_name:
            safe_name = parsed.netloc.replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save files
        files_created = []
        
        # HTML
        html_filename = os.path.join(output_dir, f"{safe_name}_{timestamp}.html")
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(result["html_content"])
        files_created.append(html_filename)
        
        # Schema
        if result.get("json_ld_schema"):
            schema_filename = os.path.join(output_dir, f"{safe_name}_schema_{timestamp}.json")
            with open(schema_filename, "w", encoding="utf-8") as f:
                f.write(result["json_ld_schema"])
            files_created.append(schema_filename)
        
        # XML Feed (individual - will be combined later)
        if result.get("feed_xml"):
            xml_filename = os.path.join(output_dir, f"{safe_name}_feed_{timestamp}.xml")
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(result["feed_xml"])
            files_created.append(xml_filename)
        
        # Crawlability Report
        if result.get("crawlability_report"):
            report_filename = os.path.join(output_dir, f"{safe_name}_crawlability_{timestamp}.json")
            with open(report_filename, "w", encoding="utf-8") as f:
                json.dump(result["crawlability_report"], f, indent=2)
            files_created.append(report_filename)
        
        print(f"\n✅ Success! Generated {len(files_created)} files")
        
        return {
            "url": url,
            "status": "success",
            "product_name": result.get("product_name"),
            "is_valid": result.get("is_valid"),
            "files": files_created,
            "errors": result.get("errors", []),
            "feed_data": result.get("feed_data")  # Return feed data for combined CSV/XML
        }
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return {
            "url": url,
            "status": "failed",
            "error": str(e),
            "files": [],
            "feed_data": None
        }
    

# Example usage
if __name__ == "__main__":
    from datetime import datetime
    import os
    import csv
    
    # Multiple product URLs to process
    product_urls = [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/the-book-of-basketball-the-nba-according-to-the-sports-guy_232/index.html",
        "https://books.toscrape.com/catalogue/sophies-world_966/index.html",
    ]
    
    # You can also load URLs from a file
    # with open('product_urls.txt', 'r') as f:
    #     product_urls = [line.strip() for line in f if line.strip()]
    
    print(f"🚀 Starting batch processing of {len(product_urls)} products")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create pipeline once
    app = create_pipeline()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_batch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all products
    results = []
    all_feed_data = []  # Collect feed data from all products
    
    for i, url in enumerate(product_urls, 1):
        print(f"\n📦 Processing product {i}/{len(product_urls)}")
        result = process_single_product(url, app, output_dir)
        results.append(result)
        
        # Collect feed data for combined feeds
        if result.get("feed_data"):
            all_feed_data.append(result["feed_data"])
    
    # Generate combined CSV feed with ALL products
    if all_feed_data:
        combined_csv_filename = os.path.join(output_dir, f"product_feed_combined_{timestamp}.csv")
        with open(combined_csv_filename, "w", encoding="utf-8", newline='') as f:
            if all_feed_data:
                fieldnames = all_feed_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_feed_data)
        print(f"\n📋 Combined CSV Feed created: {combined_csv_filename} ({len(all_feed_data)} products)")
    
    # Generate combined XML feed with ALL products
    if all_feed_data:
        combined_xml_filename = os.path.join(output_dir, f"product_feed_combined_{timestamp}.xml")
        xml_items = []
        for feed in all_feed_data:
            item_xml = f"""    <item>
      <g:id>{feed.get('id', '')}</g:id>
      <g:title><![CDATA[{feed.get('title', '')}]]></g:title>
      <g:description><![CDATA[{feed.get('description', '')}]]></g:description>
      <g:link>{feed.get('link', '')}</g:link>
      <g:image_link>{feed.get('image_link', '')}</g:image_link>
      <g:availability>{feed.get('availability', '')}</g:availability>
      <g:price>{feed.get('price', '')}</g:price>
      <g:brand>{feed.get('brand', '')}</g:brand>
      <g:condition>{feed.get('condition', 'new')}</g:condition>
    </item>"""
            xml_items.append(item_xml)
        
        combined_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
  <channel>
    <title>Product Feed</title>
    <link>{output_dir}</link>
    <description>Auto-generated product feed for {len(all_feed_data)} products</description>
    <lastBuildDate>{timestamp}</lastBuildDate>
{chr(10).join(xml_items)}
  </channel>
</rss>"""
        
        with open(combined_xml_filename, "w", encoding="utf-8") as f:
            f.write(combined_xml)
        print(f"📋 Combined XML Feed created: {combined_xml_filename} ({len(all_feed_data)} products)")
    
    # Generate summary report
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"\n✅ Successful: {len(successful)}/{len(product_urls)}")
    print(f"❌ Failed: {len(failed)}/{len(product_urls)}")
    
    if successful:
        print("\n✅ Successfully processed:")
        for r in successful:
            print(f"   • {r['product_name']} ({r['url']})")
            print(f"     Valid: {r['is_valid']}, Files: {len(r['files'])}")
    
    if failed:
        print("\n❌ Failed to process:")
        for r in failed:
            print(f"   • {r['url']}")
            print(f"     Error: {r['error']}")
    
    # Save summary to JSON
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_products": len(product_urls),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Batch processing complete!")