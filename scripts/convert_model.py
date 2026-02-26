import os
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Configuration
INPUT_MODEL_DIR = "./distilbert_subjectivity_v1"
OUTPUT_MODEL_DIR = "models/distilbert-subjectivity"

def main():
    print(f"Converting model from {INPUT_MODEL_DIR} to ONNX...")
    
    if not os.path.exists(INPUT_MODEL_DIR):
        print(f"Error: Input directory {INPUT_MODEL_DIR} does not exist.")
        return

    output_path = Path(OUTPUT_MODEL_DIR)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    try:
        model = ORTModelForSequenceClassification.from_pretrained(INPUT_MODEL_DIR, export=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)
        except Exception:
            print("Tokenizer files missing in output. Loading from base 'distilbert-base-uncased'...")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # Save to output directory
        model.save_pretrained(OUTPUT_MODEL_DIR)
        tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
        
        print(f"Success! Model exported to {OUTPUT_MODEL_DIR}")
        print("You can now select 'DistilBERT Subjectivity' in the extension.")

    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure you have installed: pip install optimum[onnxruntime]")

if __name__ == "__main__":
    main()
