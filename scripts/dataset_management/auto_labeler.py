import json
import argparse
from openai import OpenAI
import time
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Auto-label articles using a local LLM.')
    parser.add_argument('--input', type=str, default='dataset_unprocessed.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='dataset_prelabeled.jsonl', help='Output JSONL file')
    parser.add_argument('--api-base', type=str, default='http://localhost:11434/v1', help='Local LLM API Endpoint')
    parser.add_argument('--api-key', type=str, default='ollama', help='API Key (dummy for local)')
    parser.add_argument('--model', type=str, default='llama3', help='Model name to use')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')

    args = parser.parse_args()

    # Load input
    with open(args.input, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # Check for existing progress
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_ids.add(record['id'])
                except: pass
        print(f"Resuming... {len(processed_ids)} articles already processed.")

    articles_to_process = [a for a in articles if a['id'] not in processed_ids]
    print(f"Processing {len(articles_to_process)} articles...")

    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key
    )

    system_prompt = """Jesteś ekspertem od analizy mediów. Twoim zadaniem jest ocena obiektywizmu tekstu.
Oceń tekst na skali od -1.0 (całkowicie subiektywny, emocjonalny, opinia) do 1.0 (całkowicie obiektywny, suchy fakt, raport).
Zwróć TYLKO poprawny obiekt JSON w następującym formacie:
{"score": <float>, "reason": "<krótkie uzasadnienie>"}
Nie dodawaj żadnego markdownu ani tekstu przed lub po JSONie. Uzasadnienie musi być w języku polskim."""

    with open(args.output, 'a', encoding='utf-8') as out_f:
        for article in tqdm(articles_to_process):
            text_snippet = article['text'][:2000] # Limit context window if needed
            
            prompt = f"Oceń obiektywizm poniższego tekstu informacyjnego:\n\nTytuł: {article['title']}\nTreść: {text_snippet}\n..."

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                # Try to clean up if model chats too much (simple heuristic)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()

                result = json.loads(content)
                score = float(result.get('score', 0.0))
                reason = result.get('reason', '')
                
                # Clamp score
                score = max(-1.0, min(1.0, score))

                output_record = {
                    'id': article['id'],
                    'text': article['text'], # Keep full text
                    'title': article['title'],
                    'predicted_score': score,
                    'predicted_reason': reason,
                    'manual_score': None # Placeholder
                }
                
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                # print(f"Error on {article['id']}: {e}")
                # Write a failed record or just skip? 
                # Let's write with error metadata
                err_record = {
                    'id': article['id'],
                    'error': str(e)
                }
                # out_f.write(json.dumps(err_record) + "\n")
                pass 
                
    print("Done.")

if __name__ == "__main__":
    main()
