import json
import re

def clean_text(text):
    if not text:
        return ""
    
    # Normalize whitespace first to make regexes reliable at boundaries
    text = " ".join(text.split())

    # Remove "Komentarze" and everything after it
    # Note: "Komentarze" often appears followed by user comments.
    if "Komentarze" in text:
        # Heuristic: if "Komentarze" is near the end or followed by date patterns/names
        # validation showed it starts the comment section.
        text = text.split("Komentarze")[0]

    # Remove "REKLAMA"
    text = text.replace("REKLAMA", "")
    
    # Remove Metadata in brackets like [GALERIA], [FILM], [GALERIA, FILM]
    text = re.sub(r'\[[^\]]*?\]', '', text)
    
    # Remove Photo credits
    # Strategy: Remove "Fot." keyphrase. 
    # Attempt to remove the name if it is short (e.g. < 30 chars).
    # If it is long, we risk deleting text, so we only remove "Fot."
    # We look for "Fot." followed by text until a dot, but only if the match is short.
    # If the match is long, it likely merged with the next sentence.
    
    def remove_foto(match):
        content = match.group(0)
        # Typical photo credits "Fot. Name Surname." are short. 
        # Sentences are usually longer. 
        # 45 chars covers "Fot. Jan Kowalski/Agency."
        if len(content) < 45:
            return "" # Safe to remove whole credit
        else:
            # Too long, matches into text. Just remove "Fot." prefix to clean the noise.
            return content.replace("Fot.", "").strip()

    # Matches "Fot." until the next dot or end of string
    # We use a lambda to decide whether to delete the whole match or just the prefix
    text = re.sub(r'Fot\.[^.]*?(\.|$)', remove_foto, text)

    # Cleanup leftover "Fot." if it didn't match the dot pattern or was kept
    text = re.sub(r'Fot\.', '', text)

    # Remove copyright symbols and text often found at end
    text = re.sub(r'©℗.*', '', text)
    text = re.sub(r'\(PAP\)', '', text)
    text = re.sub(r'Copyright.*', '', text, flags=re.IGNORECASE)
    
    # Remove author signatures at end like (akme), (dg), (w), (k)
    # Usually at the very end.
    text = re.sub(r'\s\([a-z]{1,4}\)$', '', text)
    
    # Remove "Tekst (code)"
    text = re.sub(r'Tekst \([a-z]+\)', '', text)

    # Final whitespace cleanup
    text = " ".join(text.split())
    
    return text

def clean_title(title):
    if not title:
        return ""
    prefix = "24Kurier.pl - "
    if title.startswith(prefix):
        return title[len(prefix):]
    return title

def main():
    input_file = 'dataset_unprocessed.json'
    output_file = 'dataset_cleaned.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cleaned_data = []
        for article in data:
            original_text = article.get('text', '')
            original_title = article.get('title', '')
            
            cleaned_text = clean_text(original_text).lower()
            cleaned_title = clean_title(original_title).lower()
            
            # Skip if text becomes empty (though unlikely if it passed initial filter)
            if not cleaned_text:
                continue
                
            cleaned_article = article.copy()
            cleaned_article['text'] = cleaned_text
            cleaned_article['title'] = cleaned_title
            
            cleaned_data.append(cleaned_article)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed {len(data)} articles. Saved {len(cleaned_data)} cleaned articles to {output_file}.")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
