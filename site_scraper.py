import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time

def scrape_website(
    start_url: str,
    max_pages: int = 10,
    delay: float = 1.0
):
    """
    Simple, respectful site scraper for text collection.
    
    Args:
        start_url: starting URL to crawl
        max_pages: maximum number of pages to scrape
        delay: delay between requests (seconds)
    
    Returns:
        dict mapping URLs to extracted plain text
    """
    visited = set()
    queue = deque([start_url])
    texts = {}

    domain = urlparse(start_url).netloc

    print(f"Scraping up to {max_pages} pages from {domain} ...")

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "EducationalRAGBot/1.0"})
            if "text/html" not in response.headers.get("content-type", ""):
                continue
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract content only from the element with id="content"
        content = soup.find(id="bodyContent")
        if not content:
            print(f"⚠️ No #content section found on {url}")
            continue

        paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
        page_text = "\n".join(paragraphs)

        if page_text:
            texts[url] = page_text
            print(f"✓ Scraped: {url} ({len(page_text)} chars)")

        # Collect same-domain links
        for link in content.find_all("a", href=True):
            href = urljoin(url, link["href"])
            parsed = urlparse(href)
            if parsed.netloc == domain and href not in visited:
                if "#" not in parsed.path and href.startswith("http"):
                    queue.append(href)

        time.sleep(delay)

    print(f"\nDone. Scraped {len(texts)} pages.")
    return texts

if __name__ == "__main__":
    start = "https://wiki.tfes.org/General_Physics"
    data = scrape_website(start_url=start, max_pages=15)
    
    # Save to file
    with open("The_Flat_Earth_Wiki.txt", "w", encoding="utf-8") as f:
        for url, text in data.items():
            f.write(f"\n\n=== {url} ===\n\n{text}")
