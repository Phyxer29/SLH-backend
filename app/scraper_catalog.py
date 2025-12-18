import cloudscraper
from bs4 import BeautifulSoup
import re
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATA_PATH = "data/assessments.json"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
FALLBACK_PATH = "../bkcd/ass.json"

def scrape_catalog():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 370:
            print(f"Data verified: {len(data)} assessments loaded.")
            return
        else:
            print(f"Data incomplete ({len(data)} items). Re-scraping...")

    print("Starting SHL Catalog Crawl...")
    
    scraper = cloudscraper.create_scraper()
    links = set()
    
    try:
        resp = scraper.get(CATALOG_URL, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for a in soup.find_all("a", href=True):
            if "/product-catalog/view/" in a['href']:
                href = a['href']
            
            # FIX: Ensure we have the full canonical domain and path
                if href.startswith("http"):
                    full_url = href
                else:
                    full_url = f"https://www.shl.com{href}"
            
            # FIX: Force '/solutions' prefix if missing
                if "/products/product-catalog" in full_url and "/solutions/products" not in full_url:
                    full_url = full_url.replace("/products/product-catalog", "/solutions/products/product-catalog")
                
                links.add(full_url)
        
        if len(links) < 50:
            print(f"Only found {len(links)} links, trying fallback...")
            raise Exception("Insufficient links")
            
    except Exception as e:
        print(f"Scraping failed: {e}")
        if os.path.exists(FALLBACK_PATH):
            print(f"Using fallback data from {FALLBACK_PATH}")
            with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
                fallback_data = json.load(f)
            os.makedirs("data", exist_ok=True)
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
            print(f"Loaded {len(fallback_data)} assessments from fallback")
            return
        else:
            print("No fallback data available")
            return
    
    print(f"Collected {len(links)} product links")
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_parse_page, url): url for url in links}
        for future in tqdm(as_completed(futures), total=len(links)):
            res = future.result()
            if res:
                results.append(res)
    
    print(f"Scraped {len(results)} assessments")
    os.makedirs("data", exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} assessments")

def _parse_page(url):
    try:
        scraper = cloudscraper.create_scraper()
        resp = scraper.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        lower_t = text.lower()
        
        name_tag = soup.find("h1")
        if not name_tag:
            return None
        name = name_tag.text.strip()
        
        duration = 0
        dur_match = re.search(r'(\d+)\s*(?:min|minute)', lower_t)
        if dur_match:
            duration = int(dur_match.group(1))
        
        test_type = []
        if any(x in lower_t for x in ["java", "python", "sql", "coding", "technical", "skill"]):
            test_type.append("Knowledge & Skills")
        if any(x in lower_t for x in ["personality", "behavior", "leadership", "opq"]):
            test_type.append("Personality & Behavior")
        if not test_type:
            test_type = ["General Ability"]
        
        return {
            "name": name,
            "url": url,
            "description": text[:800],
            "duration": duration,
            "remote_support": "Yes" if "remote" in lower_t else "No",
            "adaptive_support": "Yes" if "adaptive" in lower_t else "No",
            "test_type": test_type
        }
    except:
        return None

if __name__ == "__main__":
    scrape_catalog()
