import json
import glob
import os
import hashlib

def clean_text(text):
    if not text:
        return ""
    # Normalize whitespace
    return " ".join(text.split())

def main():
    input_pattern = 'scraper/scrapper/*.json'
    output_file = 'dataset_unprocessed.json'
    
    files = glob.glob(input_pattern)
    print(f"Found {len(files)} JSON files to process.")
    
    seen_hashes = set()
    all_articles = []
    
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for article in data:
                text = article.get('text', '')
                if not text or len(text) < 100:
                    continue
                
                # Deduplication based on content hash
                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if content_hash in seen_hashes:
                    continue
                
                seen_hashes.add(content_hash)
                
                # Clean text
                cleaned_text = clean_text(text)
                
                all_articles.append({
                    'id': content_hash,
                    'title': article.get('title'),
                    'url': article.get('url'),
                    'text': cleaned_text,
                    'meta_date': article.get('date_scraped')
                })
                
        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    print(f"Consolidated {len(all_articles)} unique articles.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
