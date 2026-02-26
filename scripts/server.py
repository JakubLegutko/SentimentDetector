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
DEBERTA_LOCAL_PATH = os.path.join(PROJECT_ROOT, "models", "deberta_objectivity")
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "1dcnn_objectivity_model.pt")
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M" 

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
try:
    from train_1dcnn import Text1DCNN
except ImportError:
    Text1DCNN = None
import re
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("Loading models... this may take a while.")

# 1. Fine-Tuned DeBERTa Model
try:
    if os.path.exists(DEBERTA_LOCAL_PATH):
        print(f"Loading local DeBERTa model from {DEBERTA_LOCAL_PATH}...")
        deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_LOCAL_PATH)
        deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_LOCAL_PATH)
        deberta_classifier = pipeline("text-classification", model=deberta_model, tokenizer=deberta_tokenizer, top_k=None)
        print("Fine-Tuned DeBERTa model loaded.")
    else:
        print(f"Local model not found at {DEBERTA_LOCAL_PATH}.")
        deberta_classifier = None
except Exception as e:
    print(f"Error loading DeBERTa model: {e}")
    deberta_classifier = None

# 2. 1DCNN Objectivity Model
try:
    print(f"Loading 1DCNN model from {CNN_MODEL_PATH}...")
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu'))
    cnn_vocab = checkpoint['vocab']
    
    cnn_model = Text1DCNN(
        vocab_size=len(cnn_vocab),
        embedding_dim=checkpoint['embedding_dim'],
        num_filters=checkpoint['num_filters'],
        filter_sizes=checkpoint['filter_sizes'],
        output_dim=1,
        dropout=0.0
    )
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.eval()
    print("1DCNN Objectivity model loaded.")
except Exception as e:
    print(f"Error loading 1DCNN model: {e}")
    cnn_model = None
    cnn_vocab = None


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
    model: str = "deberta_objectivity" # Default
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
            "deberta_objectivity": deberta_classifier is not None,
            "1dcnn_objectivity_model": cnn_model is not None,
            "translation": translation_pipeline is not None
        }
    }

@app.post("/analyze")
def analyze(request: TextRequest):
    selected_model = request.model
    
    # Truncate text to avoid token limits crashes if truncation=True doesn't catch it perfectly
    # 3000 chars is roughly 600-800 tokens. Safe for 1024 models, slightly over for 512 but better.
    truncated_text = request.text[:3000] 

    if selected_model == "deberta_objectivity":
        if not deberta_classifier:
            raise HTTPException(status_code=503, detail="DeBERTa model not available")
        # DeBERTa max is 512. We enable truncation.
        results = deberta_classifier(truncated_text, truncation=True, max_length=512)
        
        # Results is usually [[{'label': 'LABEL_0', 'score': 0.123}]] due to top_k=None or single output
        score_val = 0.0
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                 score_val = results[0][0].get('score', 0)
            else:
                 score_val = results[0].get('score', 0)
        elif isinstance(results, dict):
            score_val = results.get('score', 0)
            
        return {
            "label": "Objective" if score_val > 0 else "Subjective",
            "score": score_val,
            "model": "DeBERTa Objectivity"
        }
        
    elif selected_model == "1dcnn_objectivity_model":
        if not cnn_model:
            raise HTTPException(status_code=503, detail="1DCNN model not available")
        
        # Tokenize and encode
        tokens = re.findall(r'\b\S+\b', request.text.lower())
        encoded = [cnn_vocab.get(word, cnn_vocab['<UNK>']) for word in tokens]
        
        # Pad sequence horizontally if it's too short for max filter size
        min_seq_len = 4 # Default from training script max filter size
        if len(encoded) < min_seq_len:
            encoded = encoded + [0] * (min_seq_len - len(encoded))
            
        tensor = torch.tensor([encoded], dtype=torch.long)
        
        with torch.no_grad():
            prediction = cnn_model(tensor).item()
            
        return {
            "label": "Objective" if prediction > 0 else "Subjective",
            "score": prediction,
            "model": "1DCNN Objectivity"
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
    
            print(f"Gemini API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Gemini Analysis Failed: {str(e)}")

    elif selected_model == "local_llm":
        import requests
        try:
            # Re-use Gemini Prompt Structure for consistency
            prompt = f"""You are an objectivity analyzer. Here is the text to analyze:
{request.text}

Provide the response as a score, normalized from -1 (subjective) to 1 (objective) and provide a short explanation for reasoning behind the analysis. Keep the reasoning explanation to a minimum.

You must return the response in strict JSON format:
{{
  "score": <float between -1 and 1>,
  "explanation": "<string explanation>"
}}
"""
            llm_payload = {
                "model": "deepseek", # Default to deepseek or whatever is loaded, server usually ignores if unique
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 4096
            }
            
            # Forward to local LLM server
            response = requests.post("http://localhost:11434/v1/chat/completions", json=llm_payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Basic cleanup (similar to auto_labeler)
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            output = json.loads(content)
            
            return {
                "label": "Objective" if output["score"] > 0 else "Subjective",
                "score": output["score"],
                "explanation": output["explanation"],
                "model": "Local LLM"
            }
            
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Local LLM Server not reachable (is it running on port 11434?)")
        except Exception as e:
            print(f"Local LLM Error: {e}")
            raise HTTPException(status_code=500, detail=f"Local LLM Failed: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {selected_model}")

    # We have already returned for deberta_objectivity and 1dcnn_objectivity_model
    # The below is only for fallbacks if any were here, but we return early.
    raise HTTPException(status_code=400, detail=f"Unexpected flow for model: {selected_model}")

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
