import json
import argparse
import requests
import sys
import os
import time
import re

class LLMJudge:
    def __init__(self, api_url, model_name="bielik"):
        self.api_url = api_url
        self.model_name = model_name

    def judge_record(self, text, model_responses):
        """
        Asks the LLM to judge the best response.
        Returns a dict with 'best_model' and 'choice_reason'.
        """
        
        
        # specific prompt construction - improved for clarity
        system_prompt = (
            "Jesteś obiektywnym i krytycznym sędzią. Twoim zadaniem jest ocena wielu analiz tekstu przez AI "
            "i wybranie tej, która jest najdokładniejsza, najbardziej obiektywna i najlepiej uzasadniona. "
            "Musisz podać swoją decyzję w ścisłym formacie JSON."
        )

        user_prompt = f"TEKST DO ANALIZY:\n{text}\n\n"
        user_prompt += "DOSTĘPNE OCENY MODELI:\n"
        
        available_models_list = []
        for model_name, response in model_responses.items():
            user_prompt += f"MODEL: {model_name}\n"
            user_prompt += f"Ocena: {response.get('predicted_score')}\n"
            user_prompt += f"Uzasadnienie: {response.get('predicted_reason')}\n"
            user_prompt += "---\n"
            available_models_list.append(model_name)

        user_prompt += (
            "INSTRUKCJE:\n"
            "1. Porównaj oceny.\n"
            "2. Wybierz najlepszą na podstawie tego, jak dobrze uzasadnienie pasuje do oceny i tekstu.\n"
            "3. Zwróć TYLKO poprawny obiekt JSON z następującymi polami:\n"
            "   - \"best_model\": (string) nazwa wybranego modelu (musi dokładnie pasować do jednej z nazw modeli powyżej)\n"
            "   - \"choice_reason\": (string) krótkie wyjaśnienie, dlaczego wybrano tę ocenę (w języku polskim)\n"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"} # Try to enforce JSON mode if supported
        }

        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Remove <think> tags if present
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

                # Simple cleanup to find JSON if model chats around it
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass # Fall through to fallback
                
                # Fallback extraction logic...
                # (Same as before)
                
                # ... (re-implement fallback here or structure to avoid code dup, but for now inline is fine or return None to loop?) 
                # Actually let's just use the robust fallback here too.
                
                print(f"Warning: Could not parse JSON. Attempting fallback extraction. Content end:\n...{content[-500:]}", file=sys.stderr)
                model_matches = re.findall(r'[\"\']?best_model[\"\']?\s*:\s*[\"\']([^\"\']+)[\"\']', content)
                if model_matches:
                    best_model_candidate = model_matches[-1]
                    print(f"  -> Successfully extracted best_model: {best_model_candidate}", file=sys.stderr)
                    return {
                        "best_model": best_model_candidate,
                        "choice_reason": "Response was truncated or malformed. Extracted via regex."
                    }
                
                return None # Failed this attempt, but code structure returns None anyway, maybe retry logic should only handle exceptions?
                # If we got a response but couldn't parse it, retrying likely won't help unless temp > 0. 
                # But if it's a server error, we retry.

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 500:
                    print(f"Server error 500. Retrying ({attempt+1}/{retries})...", file=sys.stderr)
                    time.sleep(2)
                    continue
                else:
                    print(f"Error calling LLM: {e}", file=sys.stderr)
                    return None
            except Exception as e:
                print(f"Error calling LLM: {e}", file=sys.stderr)
                return None
        
        return None

def process_judgement(input_file: str, api_url: str, output_file: str = None, model_name: str = "bielik"):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    judge = LLMJudge(api_url, model_name)
    judged_count = 0
    total_count = len(data)

    # Determine output file name early
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_judged_{model_name}.json"

    print(f"Processing {total_count} records with LLM Judge ({model_name})...")
    print(f"Output will be saved to: {output_file}")

    try:
        for i, record in enumerate(data):
            if 'model_responses' in record and record['model_responses'] and 'text' in record and record['text']:
                
                # Check if already judged (optional optimization, but good for resuming)
                if 'judge_selected_model' in record:
                    print(f"[{i+1}/{total_count}] Skipping ID: {record.get('id')} (Already judged)")
                    judged_count += 1
                    continue

                print(f"[{i+1}/{total_count}] Judging ID: {record.get('id')}...")
                
                judgment = judge.judge_record(record['text'], record['model_responses'])
                
                if judgment and 'best_model' in judgment:
                    best_model = judgment['best_model']
                    
                    # Normalize model name if needed (e.g. if model hallucinates slightly different casing)
                    # We try to match keys
                    available_models = list(record['model_responses'].keys())
                    
                    selected_key = None
                    if best_model in available_models:
                        selected_key = best_model
                    else:
                        # Try case insensitive
                        for k in available_models:
                            if k.lower() == best_model.lower():
                                selected_key = k
                                break
                    
                    if selected_key:
                        best_response = record['model_responses'][selected_key]
                        record['predicted_score'] = best_response.get('predicted_score')
                        record['predicted_reason'] = best_response.get('predicted_reason')
                        record['choice_reason'] = judgment.get('choice_reason', "No reason provided.")
                        record['judge_selected_model'] = selected_key
                        judged_count += 1
                        print(f"  -> Selected {selected_key}")
                        
                        # Save after each record
                        temp_file = output_file + ".tmp"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        os.replace(temp_file, output_file)
                        
                    else:
                         print(f"  -> Error: Model '{best_model}' not found in available responses {available_models}")
                else:
                    print(f"  -> Failed to get valid judgment (Model: {model_name})")
            else:
                # Skip if no text or no responses
                pass
                    
    except KeyboardInterrupt:
        print("\nProcess interrupted. Data is safe.")
        
    print(f"Done. Judged {judged_count} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use LLM as a judge to select best ratings.")
    parser.add_argument("input_file", help="Input JSON file (e.g., average_review_no_score.json)")
    parser.add_argument("--api_url", default="http://localhost:11434/v1/chat/completions", help="URL of the local LLM server")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--model", default="bielik", help="Model name to use as a judge (default: bielik)")

    args = parser.parse_args()
    
    process_judgement(args.input_file, args.api_url, args.output, args.model)
