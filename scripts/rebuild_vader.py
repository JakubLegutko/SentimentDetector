
import json
import urllib.request
import re
import os

LEXICON_URL = "https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt"
VADER_JS_PATH = os.path.join(os.path.dirname(__file__), "../vendor/vader.js")

def rebuild_vader():
    print(f"Downloading lexicon from {LEXICON_URL}...")
    try:
        with urllib.request.urlopen(LEXICON_URL) as response:
            data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download lexicon: {e}")
        return

    print("Parsing lexicon...")
    lexicon_js_lines = []
    lexicon_js_lines.append("  const LEXICON = {")
    
    count = 0
    lines = data.strip().split('\n')
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            token = parts[0]
            measure = parts[1]
            # Use json.dumps to handle specific characters like backslashes and quotes safely
            token_safe = json.dumps(token) 
            # token_safe includes quotes, e.g. "token"
            lexicon_js_lines.append(f"    {token_safe}: {measure},")
            count += 1
            
    lexicon_js_lines.append("  }")
    full_lexicon_js = "\n".join(lexicon_js_lines)
    print(f"Parsed {count} words.")

    print(f"Reading {VADER_JS_PATH}...")
    try:
        with open(VADER_JS_PATH, 'r', encoding='utf-8') as f:
            js_content = f.read()
    except FileNotFoundError:
        print(f"vader.js not found at {VADER_JS_PATH}")
        return

    # Replace the LEXICON object
    # We look for "const LEXICON = {" and the closing "  }"
    # Since the file might be truncated or formatted strictly, we use regex
    # Pattern: const LEXICON = \{ .*? \n  \}
    
    pattern = re.compile(r"const LEXICON = \{[\s\S]*?^\s\s\}", re.MULTILINE)
    match = pattern.search(js_content)
    
    if match:
        print("Found LEXICON object. Replacing...")
        # Use string slicing to avoid re.sub escaping issues with the massive lexicon content
        new_js_content = js_content[:match.start()] + full_lexicon_js + js_content[match.end():]
        
        with open(VADER_JS_PATH, 'w', encoding='utf-8') as f:
            f.write(new_js_content)
        print("Successfully rebuilt vendor/vader.js with full lexicon.")
    else:
        print("Could not find LEXICON object in vader.js via regex. Dumping structure for debug.")
        print(js_content[:500])

if __name__ == "__main__":
    rebuild_vader()
