import json
import argparse
import sys
import os
import time
import requests
import datetime
from readability import Document
from bs4 import BeautifulSoup

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ArticleRegenerator:
    def __init__(self):
        # Use requests directly to avoid SSL context issues with cloudscraper when verify=False is needed
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_text(self, url):
        try:
            response = self.session.get(url, timeout=15, verify=False)
            response.raise_for_status()
            
            doc = Document(response.text)
            content_html = doc.summary()
            
            soup = BeautifulSoup(content_html, 'lxml')
            text_content = soup.get_text(separator='\n\n', strip=True)
            
            if not text_content or len(text_content) < 100: 
                return None

            return text_content
            
        except Exception as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            return None

def regenerate_dataset(input_file: str):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    regenerator = ArticleRegenerator()
    recreated_count = 0
    total_count = len(data)
    
    print(f"Processing {total_count} records...")

    # We iterate and modify in place
    try:
        for i, record in enumerate(data):
            if 'url' in record and record['url']:
                print(f"[{i+1}/{total_count}] Fetching: {record['url']}")
                text = regenerator.fetch_text(record['url'])
                
                if text:
                    record['text'] = text
                    recreated_count += 1
                else:
                    print(f"Failed to fetch/extract text for {record['url']}")
                    if 'text' not in record:
                        record['text'] = None
            else:
                 if 'text' not in record:
                        record['text'] = None
            
            # Be polite to servers
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")

    base_name = os.path.splitext(input_file)[0]
    if base_name.endswith("_trim"):
        output_file = base_name.replace("_trim", "_recreated") + ".json"
    else:
        output_file = base_name + "_recreated.json"
        
    print(f"Saving {len(data)} records to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Recreated text for {recreated_count} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate text content from URLs in a trimmed dataset.")
    parser.add_argument("input_file", help="Input JSON file (e.g., average_review_trim.json)")
    
    args = parser.parse_args()
    
    regenerate_dataset(args.input_file)
