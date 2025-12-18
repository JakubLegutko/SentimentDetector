import requests
import cloudscraper
from readability import Document
from bs4 import BeautifulSoup
import datetime
import argparse
import sys
import json
import os
from urllib.parse import urljoin, urlparse
import time

class ArticleScraper:
    def __init__(self):
        # Use cloudscraper to bypass Cloudflare
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_article(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            doc = Document(response.text)
            title = doc.title()
            content_html = doc.summary()
            
            soup = BeautifulSoup(content_html, 'lxml')
            text_content = soup.get_text(separator='\n\n', strip=True)
            
            if not text_content or len(text_content) < 100: 
                return None

            date_scraped = datetime.datetime.now().isoformat()
            
            return {
                'title': title,
                'text': text_content,
                'url': url,
                'date_scraped': date_scraped
            }
            
        except Exception as e:
            # print(f"Error parsing {url}: {e}", file=sys.stderr) 
            return None

    def extract_links(self, url):
        """Fetches a page and returns valid internal article links."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            base_domain = urlparse(url).netloc
            links = set()

            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                parsed_url = urlparse(full_url)
                
                # Filter for internal links only
                if parsed_url.netloc == base_domain:
                    # Heuristic: Article links usually have a longer path
                    if len(parsed_url.path) > 15: 
                         links.add(full_url)
            
            return list(links)
        except Exception as e:
            print(f"Error extracting links from {url}: {e}", file=sys.stderr)
            return []
    
    def extract_external_links_from_html(self, html_content, base_url=""):
        """Parses HTML content and returns list of external news site URLs."""
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            sites = set()
            ignore_domains = {'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com', 'wikipedia.org', 'google.com', 'linkedin.com', 'kadaza.pl', 'kadaza.com'}

            for a in soup.find_all('a', href=True):
                href = a['href']
                if base_url:
                    full_url = urljoin(base_url, href)
                else:
                    full_url = href # Contain raw href if no base
                
                parsed_url = urlparse(full_url)
                
                # Basic validation that it looks like a url
                if not parsed_url.scheme or not parsed_url.netloc:
                    continue

                # Basic filtering of social media and garbage
                domain_parts = parsed_url.netloc.split('.')
                root_domain = ".".join(domain_parts[-2:]) if len(domain_parts) >= 2 else parsed_url.netloc
                
                if root_domain not in ignore_domains:
                        # Normalize to homepage
                        homepage = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        sites.add(homepage)
            
            return list(sites)
        except Exception as e:
            print(f"Error extracting sites from content: {e}", file=sys.stderr)
            return []

    def crawl(self, start_url, limit=10):
        try:
            print(f"Crawling {start_url}...", file=sys.stderr)
            links = self.extract_links(start_url)
            print(f"Found {len(links)} potential articles on {start_url}. Scraping first {limit}...", file=sys.stderr)
            
            results = []
            count = 0
            for link in links:
                if count >= limit:
                    break
                
                if link == start_url:
                    continue

                article = self.fetch_article(link)
                if article:
                    results.append(article)
                    count += 1
                    print(f"[{count}/{limit}] Scraped: {article['title']}", file=sys.stderr)
            
            return results
        except Exception as e:
             print(f"Error crawling {start_url}: {e}", file=sys.stderr)
             return []

def main():
    parser = argparse.ArgumentParser(description='Scrape articles from Polish websites.')
    parser.add_argument('input', metavar='INPUT', type=str, nargs='*', help='URLs to scrape/crawl or file path for bulk mode')
    parser.add_argument('--crawl', action='store_true', help='Crawl the given URL for articles')
    parser.add_argument('--bulk', action='store_true', help='Treat input as a source list (URL or local file) and crawl linked sites recursively')
    parser.add_argument('--limit', type=int, default=10, help='Max articles to scrape per site')
    parser.add_argument('--output', type=str, help='Output JSON file path (ignored in bulk mode)')
    
    args = parser.parse_args()

    scraper = ArticleScraper()
    
    inputs = args.input
    if not inputs:
        print("Enter URLs/Path (one per line, Ctrl+D or Ctrl+Z to finish):", file=sys.stderr)
        inputs = sys.stdin.read().splitlines()

    # Create scraped data directory for bulk output if needed
    if args.bulk:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # scripts/scraper -> scripts -> root
        output_dir = os.path.join(base_dir, 'datasets', 'scraped')
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    
    for item in inputs:
        item = item.strip()
        if not item:
            continue
            
        if args.bulk:
            print(f"=== Starting BULK scrape from {item} ===", file=sys.stderr)
            
            # Check if it's a file
            if os.path.isfile(item):
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        content = f.read()
                    sites = scraper.extract_external_links_from_html(content)
                except Exception as e:
                    print(f"Error reading file {item}: {e}", file=sys.stderr)
                    continue
            else:
                 # Assume it's a URL
                 try:
                    print(f"Fetching source list from {item}...", file=sys.stderr)
                    response = scraper.session.get(item, timeout=20)
                    response.raise_for_status()
                    sites = scraper.extract_external_links_from_html(response.text, base_url=item)
                 except Exception as e:
                     print(f"Error fetching source {item}: {e}", file=sys.stderr)
                     continue

            print(f"Found {len(sites)} news sites to scrape.", file=sys.stderr)
            
            for site_url in sites:
                domain = urlparse(site_url).netloc.replace('www.', '')
                filename = os.path.join(output_dir, f"articles_{domain}.json")
                print(f"--- Processing {site_url} -> {filename} ---", file=sys.stderr)
                
                site_results = scraper.crawl(site_url, limit=args.limit)
                
                if site_results:
                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(site_results, f, ensure_ascii=False, indent=2)
                        print(f"Saved {len(site_results)} articles to {filename}", file=sys.stderr)
                    except Exception as e:
                        print(f"Failed to save {filename}: {e}", file=sys.stderr)
                else:
                    print(f"No articles found for {site_url}", file=sys.stderr)
                    
        elif args.crawl:
            results = scraper.crawl(item, limit=args.limit)
            all_results.extend(results)
        else:
            print(f"Scraping: {item}...", file=sys.stderr)
            article = scraper.fetch_article(item)
            if article:
                all_results.append(article)
    
    # Only manage global output if NOT in bulk mode (bulk mode writes individual files)
    if not args.bulk:
        output_json = json.dumps(all_results, ensure_ascii=False, indent=2)
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_json)
                print(f"Saved {len(all_results)} articles to {args.output}", file=sys.stderr)
            except Exception as e:
                print(f"Error saving to file: {e}", file=sys.stderr)
                print(output_json)
        else:
            print(output_json)

if __name__ == "__main__":
    main()
