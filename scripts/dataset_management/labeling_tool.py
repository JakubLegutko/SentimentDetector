import json
import os
import sys
import shutil

try:
    import colorama
    from colorama import Fore, Style, Back
    colorama.init()
except ImportError:
    class Fore: RED=GREEN=YELLOW=BLUE=CYAN=RESET=""
    class Style: BRIGHT=DIM=RESET_ALL=""
    class Back: RED=GREEN=RESET=""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def save_append(file_path, record):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

def load_processed_ids(file_path):
    ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ids.add(rec['id'])
                except: pass
    return ids

def main():
    input_file = "dataset_prelabeled.jsonl"
    output_file = "dataset_labeled_final.jsonl"
    
    if not os.path.exists(input_file):
        # Fallback to unprocessed if prelabeled doesn't exist
        input_file = "dataset_unprocessed.json"
        print(f"Prelabeled file not found, falling back to {input_file} (no AI predictions)")
        try:
             with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            print("No data found!")
            return
    else:
        # Load JSONL
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: pass

    processed_ids = load_processed_ids(output_file)
    to_process = [d for d in data if d['id'] not in processed_ids]
    
    if not to_process:
        print("All articles processed!")
        return

    print(f"Starting labeling session. {len(to_process)} articles remaining.")
    print("Controls: [1-5] to Rate, [Space] Skip, [q] Quit")
    input("Press Enter to start...")

    for i, article in enumerate(to_process):
        clear_screen()
        print(f"{Back.BLUE}{Fore.WHITE} Progress: [{i+1}/{len(to_process)}] {Style.RESET_ALL}")
        print(f"{Style.BRIGHT}Title:{Style.RESET_ALL} {article['title']}")
        print("-" * 80)
        
        # Determine predicted score color
        pred_score = article.get('predicted_score')
        pred_reason = article.get('predicted_reason', 'N/A')
        
        if pred_score is not None:
             score_color = Fore.GREEN if pred_score > 0.3 else (Fore.RED if pred_score < -0.3 else Fore.YELLOW)
             print(f"AI Prediction: {score_color}{pred_score:.2f}{Style.RESET_ALL} | Reason: {pred_reason}")
             print("-" * 80)

        # Show text snippet
        print(article.get('text', '')[:1000] + "...\n")
        print("-" * 80)
        print(f"{Style.BRIGHT}Rate Objectivity (-1.0 to 1.0):{Style.RESET_ALL}")
        print("1: -1.0 (Subjective/Opinion)")
        print("2: -0.5")
        print("3:  0.0 (Neutral/Mixed)")
        print("4:  0.5")
        print("5:  1.0 (Objective/Fact)")
        print("[Space]: Skip")
        print("[q]: Quit")
        
        while True:
            choice = input(f"{Fore.CYAN}Bypass > {Style.RESET_ALL}").strip().lower()
            
            score = None
            if choice == 'q':
                print("Exiting...")
                sys.exit(0)
            elif choice == ' ':
                break # Skip
            elif choice == '1': score = -1.0
            elif choice == '2': score = -0.5
            elif choice == '3': score = 0.0
            elif choice == '4': score = 0.5
            elif choice == '5': score = 1.0
            
            if score is not None:
                article['manual_score'] = score
                save_append(output_file, article)
                print(f"{Fore.GREEN}Saved!{Style.RESET_ALL}")
                time.sleep(0.2)
                break
            else:
                 print("Invalid input.")
        
    print("\nSession complete!")

if __name__ == "__main__":
    import time
    main()
