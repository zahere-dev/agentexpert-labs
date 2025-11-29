from typing import List
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import Counter
import re
import json
from loguru import logger

from models import AgentState


class ParserService:
    def __init__(self, timeout: int = 10, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0 Safari/537.36"
            )
        }

    def _fetch(self, url: str) -> str:
        """Fetch HTML content with retries and timeouts."""
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.text
            except Exception as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Failed to fetch {url}: {e}")
        return ""

    def _clean_text(self, html: str) -> str:
        """Remove scripts/styles and return normalized text."""
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def _extract_meta(self, soup: BeautifulSoup) -> dict:
        """Extract meta tags useful for SEO/AEO."""
        meta_data = {
            "title": soup.title.string.strip() if soup.title else None,
            "description": None,
            "keywords": None,
            "og": {},
            "twitter": {},
            "canonical": None,
            "robots": None,
            "hreflang": []
        }

        # Meta tags
        for tag in soup.find_all("meta"):
            if tag.get("name") == "description":
                meta_data["description"] = tag.get("content")
            elif tag.get("name") == "keywords":
                meta_data["keywords"] = tag.get("content")
            elif tag.get("name") == "robots":
                meta_data["robots"] = tag.get("content")

            # Open Graph
            if tag.get("property", "").startswith("og:"):
                meta_data["og"][tag["property"]] = tag.get("content")
            # Twitter
            if tag.get("name", "").startswith("twitter:"):
                meta_data["twitter"][tag["name"]] = tag.get("content")

        # Canonical
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            meta_data["canonical"] = canonical["href"]

        # Hreflangs
        hreflangs = soup.find_all("link", rel="alternate", hreflang=True)
        meta_data["hreflang"] = [link["href"] for link in hreflangs if link.get("href")]
        
        
        return meta_data
    
    def _extract_image_tags(self, soup:BeautifulSoup, base_url:str) -> List:
          # === Alt tag validation for images ===
        generic_alts = {"image", "photo", "picture", "graphic", "logo"}
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            alt = img.get("alt", "").strip()
            images.append({
                "src": urljoin(base_url, src) if src else None,
                "alt": alt if alt else None,
                "alt_missing": not bool(alt),
                "alt_generic": alt.lower() in generic_alts if alt else False
            })
        return images

    def _extract_headings(self, soup: BeautifulSoup) -> dict:
        """Extract headings structure for SEO semantic analysis."""
        headings = {}
        for level in range(1, 7):
            tags = [h.get_text(strip=True) for h in soup.find_all(f"h{level}")]
            if tags:
                headings[f"h{level}"] = tags
        return headings

    def _validate_headings(self, headings: dict) -> list:
        """Check heading SEO issues like missing/multiple H1s."""
        issues = []
        if "h1" not in headings or len(headings["h1"]) == 0:
            issues.append("Missing H1")
        if "h1" in headings and len(headings["h1"]) > 1:
            issues.append("Multiple H1s found")
        return issues

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> dict:
        """Extract internal vs external links, check broken links."""
       # === Links with anchor text ===
        domain = urlparse(base_url).netloc
        all_links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            anchor_text = a.get_text(strip=True) or None
            link_domain = urlparse(href).netloc
            is_internal = (link_domain == domain)
            all_links.append({
                "url": href,
                "text": anchor_text,
                "is_internal": is_internal
            })

        # === Header/Footer links ===
        header_links, footer_links = [], []
        if soup.header:
            header_links = [{"url": urljoin(base_url, a["href"]),
                             "text": a.get_text(strip=True) or None}
                            for a in soup.header.find_all("a", href=True)]
        if soup.footer:
            footer_links = [{"url": urljoin(base_url, a["href"]),
                             "text": a.get_text(strip=True) or None}
                            for a in soup.footer.find_all("a", href=True)]
            
            
        return {"all_links":all_links,"header_links":header_links,"footer_links":footer_links}

        # # Validate broken links (basic, HEAD requests)
        # broken_links = []
        # with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
        #     for link in list(internal)[:20] + list(external)[:20]:  # limit to 20 each for performance
        #         try:
        #             r = client.head(link)
        #             if r.status_code >= 400:
        #                 broken_links.append(link)
        #         except Exception:
        #             broken_links.append(link)

        # return {
        #     "internal_links": list(internal),
        #     "external_links": list(external),
        #     "broken_links": broken_links
        # }

    def _extract_structured_data(self, soup: BeautifulSoup) -> list:
        """Extract JSON-LD structured data for AEO."""
        jsonld_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                jsonld_data.append(json.loads(script.string))
            except Exception:
                continue
        return jsonld_data

    def _keyword_analysis(self, text: str, top_n: int = 20) -> dict:
        """Basic keyword frequency analysis for SEO."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        freq = Counter(words)
        return dict(freq.most_common(top_n))

    def _page_size_metrics(self, html: str, soup: BeautifulSoup) -> dict:
        """Basic page size and asset metrics."""
        return {
            "html_size_bytes": len(html.encode("utf-8")),
            "script_count": len(soup.find_all("script")),
            "css_count": len(soup.find_all("link", rel="stylesheet"))
        }
        
    def _extract_section_links(self,url: str,section):
        urls = []
        if section:
            for a in section.find_all("a", href=True):
                urls.append(urljoin(url, a["href"]))
        return list(set(urls))


    def scrape_url(self, url: str) -> dict:
        """
        Scrape URL, clean text, and return SEO/AEO enriched data.
        """
        html = self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        clean_text = self._clean_text(html)
        meta = self._extract_meta(soup)
        headings = self._extract_headings(soup)
        heading_issues = self._validate_headings(headings)
        links = self._extract_links(soup, url)
        structured_data = self._extract_structured_data(soup)
        keywords = self._keyword_analysis(clean_text)
        page_metrics = self._page_size_metrics(html, soup)
        images= self._extract_image_tags(soup=soup,base_url=url)
        

        return {
            "url": url,
            "meta": meta,
            "headings": headings,
            "heading_issues": heading_issues,
            "links": links,
            "images":images,
            "structured_data": structured_data,
            "word_count": len(clean_text.split()),
            "keywords": keywords,
            "page_metrics": page_metrics,
            "content": clean_text           
        }



def scraper_node(state: AgentState) -> AgentState:
    logger.info(f"✨ Scraping the webpage content for URL: {state['url']}...")
    parser = ParserService()
    url = state["url"]
    scraped_data = parser.scrape_url(url)
    state["html_content"] = scraped_data["content"]
    logger.info("✅ Webpage content scraped successfully.")
    return state
