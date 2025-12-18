from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import torch
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv() # Load env vars from .env file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, necessary for chrome-extension://
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "distilbert_subjectivity_v1")
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M" 
DEBERTA_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
BERT_MODEL_ID = "bert-base-uncased"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("Loading models... this may take a while.")

# 1. Fine-Tuned Subjectivity Model (DistilBERT)
try:
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading local subjectivity model from {LOCAL_MODEL_PATH}...")
        try:
            subjectivity_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        except Exception:
            print("Local tokenizer not found, falling back to distilbert-base-uncased tokenizer.")
            subjectivity_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
        subjectivity_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    else:
        print(f"Local model not found at {LOCAL_MODEL_PATH}. Using base distilbert (for testing only).")
        subjectivity_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        subjectivity_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    subjectivity_classifier = pipeline("text-classification", model=subjectivity_model, tokenizer=subjectivity_tokenizer, top_k=None)
    
    # Set explicit labels
    subjectivity_model.config.id2label = {0: "subjective", 1: "objective"}
    subjectivity_model.config.label2id = {"subjective": 0, "objective": 1}
    
    print("Fine-Tuned Subjectivity model loaded.")
except Exception as e:
    print(f"Error loading subjectivity model: {e}")
    subjectivity_classifier = None

# 2. DeBERTa v3 (Zero-Shot)
try:
    print(f"Loading DeBERTa model {DEBERTA_MODEL_ID}...")
    # Zero-shot classification pipeline uses an NLI model
    deberta_classifier = pipeline("zero-shot-classification", model=DEBERTA_MODEL_ID)
    print("DeBERTa model loaded.")
except Exception as e:
    print(f"Error loading DeBERTa model: {e}")
    deberta_classifier = None

# 3. BERT Base (Feature Extraction)
try:
    print(f"Loading BERT model {BERT_MODEL_ID}...")
    bert_pipeline = pipeline("feature-extraction", model=BERT_MODEL_ID)
    print("BERT model loaded.")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    bert_pipeline = None

# Load Translation Model
try:
    print(f"Loading translation model {TRANSLATION_MODEL_ID}...")
    translation_pipeline = pipeline("translation", model=TRANSLATION_MODEL_ID)
    print("Translation model loaded.")
except Exception as e:
    print(f"Error loading translation model: {e}")
    translation_pipeline = None

class TextRequest(BaseModel):
    text: str
    model: str = "distilbert-subjectivity" # Default
    api_key: str = None

class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "auto"
    tgt_lang: str = "eng_Latn"

@app.get("/")
def read_root():
    return {
        "status": "running", 
        "models": {
            "distilbert-subjectivity": subjectivity_classifier is not None,
            "deberta-zero-shot": deberta_classifier is not None,
            "bert-base": bert_pipeline is not None,
            "translation": translation_pipeline is not None
        }
    }

@app.post("/analyze")
def analyze(request: TextRequest):
    selected_model = request.model
    
    # Truncate text to avoid token limits crashes if truncation=True doesn't catch it perfectly
    # 3000 chars is roughly 600-800 tokens. Safe for 1024 models, slightly over for 512 but better.
    truncated_text = request.text[:3000] 

    if selected_model == "distilbert-subjectivity":
        if not subjectivity_classifier:
            raise HTTPException(status_code=503, detail="Fine-tuned model not available")
        # DistilBERT max is 512. We enable truncation.
        results = subjectivity_classifier(truncated_text, truncation=True, max_length=512)
        
    elif selected_model == "deberta-zero-shot":
        if not deberta_classifier:
            raise HTTPException(status_code=503, detail="DeBERTa model not available")
        # DeBERTa v3 max is 512/1024? Enable truncation.
        results = deberta_classifier(truncated_text, candidate_labels=["subjective", "objective"], truncation=True, max_length=512)
        print(f"DEBERTA RETURN: {results}")
        return results

    elif selected_model == "bert-base":
        if not bert_pipeline:
             raise HTTPException(status_code=503, detail="BERT model not available")
        return {
            "label": "Needs Fine-tuning",
            "score": 0.5,
            "model": "BERT Base (Local)"
        }
    
    elif selected_model == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or request.api_key
        if not api_key:
             raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set and no key provided in request.")
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""You are an objectivity analyzer. Here is the text to analyze:
{request.text}

Provide the response as a score, normalized from -1 (subjective) to 1 (objective) and provide a short explanation for reasoning behind the analysis. Keep the reasoning explanation to a minimum.

You must return the response in strict JSON format:
{{
  "score": <float between -1 and 1>,
  "explanation": "<string explanation>"
}}
"""
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            output = json.loads(response.text)
            
            return {
                "label": "Objective" if output["score"] > 0 else "Subjective",
                "score": output["score"], # Raw -1 to 1
                "explanation": output["explanation"],
                "model": "Gemini 1.5 Flash"
            }

        except Exception as e:
            print(f"Gemini API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Gemini Analysis Failed: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {selected_model}")

    # For Text Classification (DistilBERT)
    # Validately unwrap all nesting (e.g. [[{...}]] or [{...}])
    final_result = results
    while isinstance(final_result, list) and len(final_result) > 0 and isinstance(final_result[0], list):
        final_result = final_result[0]
    
    print(f"ANALYZE RETURN ({selected_model}): {final_result}")
    return final_result

@app.post("/translate")
def translate(request: TranslationRequest):
    if not translation_pipeline:
        raise HTTPException(status_code=503, detail="Translation model not available")
    
    src_lang = request.src_lang
    
    # Language Auto-Detection
    if src_lang == "auto" or not src_lang:
        try:
            from langdetect import detect
            detected = detect(request.text)
            # Map common ISO 639-1 codes to NLLB codes
            nllb_map = {
                'en': 'eng_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn',
                'it': 'ita_Latn', 'pt': 'por_Latn', 'pl': 'pol_Latn', 'ru': 'rus_Cyrl',
                'uk': 'ukr_Cyrl', 'ja': 'jpn_Jpan', 'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant',
                'zh': 'zho_Hans', 'nl': 'nld_Latn', 'sv': 'swe_Latn', 'da': 'dan_Latn',
                'fi': 'fin_Latn', 'no': 'nob_Latn', 'tr': 'tur_Latn', 'ar': 'arb_Arab',
                'hi': 'hin_Deva', 'ko': 'kor_Hang'
            }
            src_lang = nllb_map.get(detected, 'eng_Latn') 
            print(f"Detected language: {detected} -> {src_lang}")
        except ImportError:
            print("langdetect not installed. Defaulting src_lang to eng_Latn. Install 'langdetect' for auto-detection.")
            src_lang = 'eng_Latn'
        except Exception as e:
            print(f"Language detection failed: {e}")
            src_lang = 'eng_Latn'
    
    # Skip translation if source matches target
    if src_lang == request.tgt_lang:
        print(f"Source language matches target ({src_lang}), skipping translation.")
        return {"translated_text": request.text}

    try:
        # Increase max_length as requested by user, but enable truncation for the input
        # Note: NLLB max position embeddings is usually 1024.
        # We also truncate the input string to avoid extreme cases that might bypass tokenizer limits or cause OOM.
        safe_text_input = request.text[:5000] 
        output = translation_pipeline(safe_text_input, src_lang=src_lang, tgt_lang=request.tgt_lang, max_length=1024, truncation=True)
        # output is [{'translation_text': '...'}]
        return {"translated_text": output[0]['translation_text']}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
