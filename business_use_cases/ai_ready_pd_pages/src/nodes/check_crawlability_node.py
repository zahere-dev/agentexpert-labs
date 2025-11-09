import json
from models import ProductPageState
from bs4 import BeautifulSoup

def check_crawlability(state: ProductPageState) -> ProductPageState:
    """Agent: Check if site is crawlable by AI bots"""
    print("🤖 Checking AI crawlability...")
    
    from urllib.parse import urlparse, urljoin
    import requests
    
    report = {
        "gptbot_allowed": None,
        "robots_txt_exists": False,
        "content_in_html": False,
        "javascript_required": False,
        "recommendations": []
    }
    
    parsed_url = urlparse(state["url"])
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_url = urljoin(base_url, "/robots.txt")
    
    # Check robots.txt
    try:
        robots_response = requests.get(robots_url, timeout=5)
        if robots_response.status_code == 200:
            report["robots_txt_exists"] = True
            state["robots_txt_content"] = robots_response.text
            
            # Check for GPTBot
            robots_text = robots_response.text.lower()
            if "gptbot" in robots_text:
                if "disallow" in robots_text.split("gptbot")[1].split("\n")[0]:
                    report["gptbot_allowed"] = False
                    report["recommendations"].append("⚠️ GPTBot is disallowed in robots.txt - AI crawlers cannot access your site")
                else:
                    report["gptbot_allowed"] = True
            else:
                report["gptbot_allowed"] = True  # Not explicitly blocked
                report["recommendations"].append("✓ GPTBot not blocked (implicitly allowed)")
        else:
            report["recommendations"].append("⚠️ robots.txt not found - consider adding one")
    except Exception as e:
        print(f"Could not fetch robots.txt: {e}")
        report["recommendations"].append("⚠️ Could not access robots.txt")
    
    # Check if content is in raw HTML
    soup = BeautifulSoup(state["html_content"], 'html.parser')
    
    # Remove scripts and check if meaningful content remains
    soup_copy = BeautifulSoup(state["html_content"], 'html.parser')
    for script in soup_copy(['script', 'style']):
        script.decompose()
    
    text_content = soup_copy.get_text(strip=True)
    
    if state["product_name"] and state["product_name"] in text_content:
        report["content_in_html"] = True
        report["recommendations"].append("✓ Product content is in raw HTML (good for crawlers)")
    else:
        report["content_in_html"] = False
        report["javascript_required"] = True
        report["recommendations"].append("⚠️ Content may be JavaScript-only - consider server-side rendering (SSR)")
    
    # Check for common JS frameworks that might block crawlers
    html_lower = state["html_content"].lower()
    js_frameworks = []
    if "react" in html_lower or "reactdom" in html_lower:
        js_frameworks.append("React")
    if "vue" in html_lower:
        js_frameworks.append("Vue")
    if "angular" in html_lower:
        js_frameworks.append("Angular")
    
    if js_frameworks:
        report["recommendations"].append(f"ℹ️ Detected frameworks: {', '.join(js_frameworks)} - ensure SSR is enabled")
    
    state["crawlability_report"] = report
    
    print("✅ Crawlability check complete")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    return state