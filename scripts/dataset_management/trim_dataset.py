import json
import argparse
import os
import sys
from typing import List, Dict, Any

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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

def process_file(filepath: str, metadata_map: Dict[str, Dict[str, Any]]):
    print(f"Processing {filepath}...")
    dataset = load_dataset(filepath)
    if not dataset:
        print(f"Skipping empty or invalid file: {filepath}")
        return

    processed_count = 0
    for record in dataset:
        # Remove text field
        if 'text' in record:
            del record['text']
        
        # Add metadata if ID matches
        if 'id' in record and record['id'] in metadata_map:
            meta = metadata_map[record['id']]
            record['url'] = meta['url']
            record['meta_date'] = meta['meta_date']
            processed_count += 1
        elif 'id' not in record:
             print(f"Warning: Record without ID found in {filepath}")

    # Determine output filename
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    name_part, ext = os.path.splitext(base_name)
    output_filename = f"{name_part}_trim{ext}"
    output_path = os.path.join(dir_name, output_filename)

    print(f"Saving {len(dataset)} records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Finished processing {filepath}. matched metadata for {processed_count} records.")

def main():
    parser = argparse.ArgumentParser(description="Replace 'text' field with URL and metadata in datsets.")
    parser.add_argument("input_files", nargs='+', help="Input JSON files to process")
    parser.add_argument("--metadata", default=r"datasets\dataset_cleaned.json", help="Path to the cleaned dataset for metadata lookup")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file '{args.metadata}' not found.")
        sys.exit(1)

    metadata_map = load_metadata(args.metadata)
    
    for filepath in args.input_files:
        if os.path.exists(filepath):
            process_file(filepath, metadata_map)
        else:
            print(f"Error: Input file '{filepath}' not found.")

if __name__ == "__main__":
    main()
