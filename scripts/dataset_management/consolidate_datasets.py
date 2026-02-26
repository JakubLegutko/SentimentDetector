import json
import argparse
import sys
import os
from collections import defaultdict
from typing import List, Dict, Any

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Check if file contains individual JSON objects per line or a JSON array
            content = f.read().strip()
            if not content:
                print(f"Warning: File {filepath} is empty.")
                return []
            
            if content.startswith('['):
                return json.loads(content)
            else:
                # Assume JSON Lines format
                data = []
                for line in content.split('\n'):
                    if line.strip():
                        data.append(json.loads(line))
                return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def load_metadata(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Loads metadata (url, meta_date) from the cleaned dataset, keyed by ID."""
    print(f"Loading metadata from {filepath}...")
    try:
        dataset = load_dataset(filepath)
        metadata = {}
        for record in dataset:
            if 'id' in record:
                metadata[record['id']] = {
                    'url': record.get('url'),
                    'meta_date': record.get('meta_date')
                }
        print(f"Loaded metadata for {len(metadata)} records.")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def consolidate_datasets(input_files: List[str], average_output: str, manual_output: str, trim: bool = False, metadata_file: str = None, no_score: bool = False):
    records_by_id = defaultdict(list)
    
    # Pre-load metadata if trimming is enabled
    metadata_map = {}
    if trim:
        if metadata_file and os.path.exists(metadata_file):
            metadata_map = load_metadata(metadata_file)
        else:
            print(f"Warning: Metadata file '{metadata_file}' not found. Trimming might be incomplete (missing url/date).")

    # 1. Load all records and group by ID
    for filepath in input_files:
        print(f"Loading {filepath}...")
        
        # Extract LLM name from filename (e.g., "dataset_labeled_bielik-2.json" -> "bielik-2")
        filename = os.path.basename(filepath)
        name_part = os.path.splitext(filename)[0]
        if name_part.startswith("dataset_labeled_"):
            llm_name = name_part.replace("dataset_labeled_", "")
        else:
            llm_name = name_part
            
        dataset = load_dataset(filepath)
        for record in dataset:
            if 'id' in record:
                # Inject LLM name into the record temporarily for processing
                record['_llm_name'] = llm_name
                records_by_id[record['id']].append(record)
            else:
                print(f"Warning: Record without ID found in {filepath}. Skipping.")

    print(f"Found {len(records_by_id)} unique IDs.")

    average_review_list = []
    manual_review_list = []

    # 2. Process each ID
    for item_id, records in records_by_id.items():
        if not records:
            continue

        # Extract scores and build model responses
        scores = []
        valid_records = []
        model_responses = {}

        for r in records:
            llm_name = r.get('_llm_name', 'unknown')
            
            # Store per-LLM response
            response_entry = {
                "predicted_score": r.get('predicted_score'),
                "predicted_reason": r.get('predicted_reason')
            }
            model_responses[llm_name] = response_entry

            if 'predicted_score' in r and r['predicted_score'] is not None:
                try:
                    scores.append(float(r['predicted_score']))
                    valid_records.append(r)
                except ValueError:
                    pass
        
        # Base record construction
        # Use the first record as a base
        base_record = records[0].copy()
        
        # Remove temporary internal field
        if '_llm_name' in base_record:
            del base_record['_llm_name']
            
        # Add model responses
        base_record['model_responses'] = model_responses

        # Trimming logic
        if trim:
            # Replace text with URL and meta_date
            if 'text' in base_record:
                del base_record['text']
            
            # Look up metadata
            if item_id in metadata_map:
                base_record['url'] = metadata_map[item_id]['url']
                base_record['meta_date'] = metadata_map[item_id]['meta_date']
            else:
                pass

        if not scores:
            # No valid scores -> Manual review
            base_record['predicted_score'] = None
            if 'predicted_reason' in base_record:
                del base_record['predicted_reason']
            manual_review_list.append(base_record)
            continue

        avg_score = sum(scores) / len(scores)
        
        # 3. Check condition: average within 0.5 of INDIVIDUAL results
        is_consistent = all(abs(avg_score - s) <= 0.5 for s in scores)

        output_record = base_record
        
        if is_consistent:
            # 4a. Success case: Save to average_review
            
            if no_score:
                # If no_score is requested, we do NOT set the consolidated score/reason
                # We also remove them if they happen to exist from base record to avoid confusion
                output_record['predicted_score'] = None
                if 'predicted_reason' in output_record:
                    del output_record['predicted_reason']
            else:
                output_record['predicted_score'] = avg_score
                # Find record with score closest to average for the 'main' reason
                closest_record = min(valid_records, key=lambda r: abs(float(r['predicted_score']) - avg_score))
                
                if 'predicted_reason' in closest_record:
                    output_record['predicted_reason'] = closest_record['predicted_reason']
                else:
                    output_record['predicted_reason'] = None 

            average_review_list.append(output_record)

        else:
            # 4b. Fail case: Save to manual_review
            if 'predicted_reason' in output_record:
                del output_record['predicted_reason']
            
            manual_review_list.append(output_record)

    # 5. Save outputs
    print(f"Saving {len(average_review_list)} records to {average_output}...")
    with open(average_output, 'w', encoding='utf-8') as f:
        json.dump(average_review_list, f, ensure_ascii=False, indent=4)

    print(f"Saving {len(manual_review_list)} records to {manual_output}...")
    with open(manual_output, 'w', encoding='utf-8') as f:
        json.dump(manual_review_list, f, ensure_ascii=False, indent=4)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate labeled datasets.")
    parser.add_argument("input_files", nargs='+', help="Input JSON files to process")
    parser.add_argument("-trim", action="store_true", help="Remove text field and add url/meta_date from cleaned dataset")
    parser.add_argument("-no_score", action="store_true", help="Do not include average score and predicted reason in valid output, only per-LLM metrics")
    parser.add_argument("--metadata", default=r"datasets\dataset_cleaned.json", help="Path to the cleaned dataset for metadata lookup")
    
    args = parser.parse_args()
    
    if args.trim:
        average_output = "average_review_trim.json"
        manual_output = "manual_review_trim.json"
    elif args.no_score:
        average_output = "average_review_no_score.json"
        manual_output = "manual_review_no_score.json"
    else:
        average_output = "average_review.json"
        manual_output = "manual_review.json"
    consolidate_datasets(args.input_files, average_output, manual_output, trim=args.trim, metadata_file=args.metadata, no_score=args.no_score)
