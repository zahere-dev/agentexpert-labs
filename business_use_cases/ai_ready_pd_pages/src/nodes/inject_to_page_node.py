from bs4 import BeautifulSoup
from models import ProductPageState


def inject_to_page(state: ProductPageState) -> ProductPageState:
    """Agent 5: Inject generated content back to page and create product feeds"""
    print("💉 Injecting content to page and generating feeds...")
    
    soup = BeautifulSoup(state["html_content"], 'html.parser')
    
    # 1. Inject JSON-LD schema
    if state["json_ld_schema"]:
        schema_tag = soup.new_tag('script', type='application/ld+json')
        schema_tag.string = state["json_ld_schema"]
        if soup.head:
            soup.head.append(schema_tag)
    
    # 2. Update/create meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and state["meta_description"]:
        meta_desc['content'] = state["meta_description"]
    elif state["meta_description"]:
        new_meta = soup.new_tag('meta', attrs={'name': 'description', 'content': state["meta_description"]})
        if soup.head:
            soup.head.append(new_meta)
    
    # 3. Prepare feed data (Google Merchant Center format)
    state["feed_data"] = {
        "id": state["sku"],
        "title": state["optimized_title"],
        "description": state["optimized_description"],
        "link": state["url"],
        "image_link": state["images"][0] if state["images"] else "",
        "availability": state["availability"],
        "price": f"{state['price']} {state['currency']}",
        "brand": state["brand"],
        "condition": "new"
    }
    
    # 4. Generate XML Feed (Google Merchant Center format)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    xml_feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
  <channel>
    <title>Product Feed</title>
    <link>{state["url"]}</link>
    <description>Auto-generated product feed</description>
    <lastBuildDate>{timestamp}</lastBuildDate>
    <item>
      <g:id>{state["sku"]}</g:id>
      <g:title><![CDATA[{state["optimized_title"]}]]></g:title>
      <g:description><![CDATA[{state["optimized_description"]}]]></g:description>
      <g:link>{state["url"]}</g:link>
      <g:image_link>{state["images"][0] if state["images"] else ""}</g:image_link>
      <g:availability>{state["availability"]}</g:availability>
      <g:price>{state["price"]} {state["currency"]}</g:price>
      <g:brand>{state["brand"]}</g:brand>
      <g:condition>new</g:condition>
    </item>
  </channel>
</rss>"""
    
    state["feed_xml"] = xml_feed
    
    # 5. Generate CSV Feed
    import csv
    from io import StringIO
    
    csv_buffer = StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=state["feed_data"].keys())
    csv_writer.writeheader()
    csv_writer.writerow(state["feed_data"])
    state["feed_csv"] = csv_buffer.getvalue()
    
    state["html_content"] = str(soup)
    state["injection_complete"] = True
    
    print("✅ Injection complete")
    print(f"✅ XML feed generated ({len(xml_feed)} bytes)")
    print(f"✅ CSV feed generated")
    return state