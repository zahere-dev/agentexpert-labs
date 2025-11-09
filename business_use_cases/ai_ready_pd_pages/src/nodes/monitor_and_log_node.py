from models import ProductPageState


def monitor_and_log(state: ProductPageState) -> ProductPageState:
    """Final monitoring and logging"""
    print("\n" + "="*70)
    print("📊 PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n📦 Product: {state['product_name']}")
    print(f"💰 Price: {state['price']} ({state['currency']})")
    print(f"🏷️  Brand: {state['brand']}")
    print(f"✅ Valid: {state['is_valid']}")
    print(f"💉 Injected: {state['injection_complete']}")
    
    if state['errors']:
        print(f"\n⚠️  Errors: {state['errors']}")
    
    print(f"\n🎯 Optimized Title: {state.get('optimized_title', 'N/A')}")
    print(f"🔑 Keywords: {', '.join(state.get('conversational_keywords', []))}")
    
    # Crawlability report
    if state.get('crawlability_report'):
        print(f"\n🤖 CRAWLABILITY REPORT:")
        report = state['crawlability_report']
        print(f"   GPTBot Allowed: {report.get('gptbot_allowed', 'Unknown')}")
        print(f"   Content in HTML: {report.get('content_in_html', False)}")
        print(f"   JavaScript Required: {report.get('javascript_required', False)}")
        print(f"\n   Recommendations:")
        for rec in report.get('recommendations', []):
            print(f"   {rec}")
    
    # Feed info
    if state.get('feed_xml'):
        print(f"\n📋 PRODUCT FEEDS GENERATED:")
        print(f"   ✓ XML Feed: {len(state['feed_xml'])} bytes")
        print(f"   ✓ CSV Feed: {len(state['feed_csv'])} bytes")
    
    return state