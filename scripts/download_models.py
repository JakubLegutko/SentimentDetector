import os
import requests
from pathlib import Path

# Configuration
MODELS = {
    "Xenova/nllb-200-distilled-600M": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ],
        "onnx": [
            "encoder_model_quantized.onnx",
            "decoder_model_merged_quantized.onnx",
        ]
    },
    "Xenova/nli-deberta-v3-base": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "onnx": [
            "model_quantized.onnx",
        ]
    },
    "Xenova/bert-base-uncased": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "onnx": [
            "model_quantized.onnx",
        ]
    }
}

OUTPUT_BASE_DIR = Path("models")

def download_file(base_url, filename, dest_path):
    print(f"Downloading {filename}...")
    try:
        if dest_path.exists():
             print(f"Skipping {dest_path.name} (already exists)")
             return

        # Try root first
        url = f"{base_url}/{filename}"
        response = requests.get(url, stream=True)
        
        # If 404, try onnx/ subdirectory
        if response.status_code == 404:
            print(f"File not found at root, trying onnx/ subdirectory...")
            url = f"{base_url}/onnx/{filename}"
            response = requests.get(url, stream=True)

        response.raise_for_status()
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {dest_path}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    if not OUTPUT_BASE_DIR.exists():
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    for model_id, config in MODELS.items():
        print(f"\nProcessing {model_id}...")
        base_url = f"https://huggingface.co/{model_id}/resolve/main"
        model_dir = OUTPUT_BASE_DIR / model_id
        
        # Download root files
        for filename in config["files"]:
            dest_path = model_dir / filename
            download_file(base_url, filename, dest_path)
            
        # Download ONNX files
        for filename in config["onnx"]:
            dest_path = model_dir / "onnx" / filename
            download_file(base_url, filename, dest_path)
    
    print("\nDownload complete.")
    print(f"Model files are located in: {OUTPUT_BASE_DIR.resolve()}")

if __name__ == "__main__":
    main()
