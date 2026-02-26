import argparse
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import json
from huggingface_hub import hf_hub_download

# Fix for Windows: Add CUDA DLLs to search path explicitly
# This solves "FileNotFoundError: Could not find module ... llama.dll"
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
if os.path.exists(cuda_path):
    print(f"Adding CUDA DLLs from: {cuda_path}")
    try:
        os.add_dll_directory(cuda_path)
    except Exception:
        pass # Not available on older Pythons, fallback to PATH
    os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    print(f"WARNING: CUDA binaries not found at {cuda_path}. GPU offloading might fail.")

# Try to import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Please install it with CUDA support for your GTX 1080:")
    print("Example (for CUDA 12): pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    print("Or see: https://github.com/abetlen/llama-cpp-python")
    exit(1)

# Define available models (GGUF format for GTX 1080 8GB)
# We use Q4_K_M quantization which is a good balance of size/speed/perplexity
MODEL_REGISTRY = {
    "bielik": {
        "repo": "speakleash/Bielik-11B-v3.0-Instruct-GGUF",
        "filename": "Bielik-11B-v3.0-Instruct.Q4_K_M.gguf",
        "description": "Bielik 11B v3.0 - Specialized for Polish."
    },
    "llama3": {
        "repo": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "description": "Meta Llama 3 8B Instruct - Strong general capability."
    },
    "mistral": {
        "repo": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "filename": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "description": "OpenHermes 2.5 (Mistral 7B) - Excellent instruction following."
    },
    "deepseek": {
        "repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "description": "DeepSeek R1 Distill Llama 8B - Strong reasoning model."
    }
}
MODELS = MODEL_REGISTRY

app = FastAPI(title="Local LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm = None
loaded_model_name = None

# Pydantic models for OpenAI API compatibility
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    # Add other params as needed but these are the main ones

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global llm
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Adapt parameters for llama-cpp
    # llama_cpp.create_chat_completion handles message formatting using the model's chat template
    
    formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    try:
        response = llm.create_chat_completion(
            messages=formatted_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 512, # Default if not provided
            stop=request.stop,
            stream=request.stream
        )
        return response
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "local"} for k in MODELS.keys()
        ]
    }

def download_model_if_needed(model_key):
    info = MODELS[model_key]
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "gguf")
    os.makedirs(models_dir, exist_ok=True)
    
    expected_path = os.path.join(models_dir, info['filename'])
    
    if os.path.exists(expected_path):
        print(f"Model found at: {expected_path}")
        return expected_path
    
    print(f"Model not found. Downloading {info['filename']} from {info['repo']}...")
    print("This may take a while...")
    
    try:
        file_path = hf_hub_download(
            repo_id=info['repo'],
            filename=info['filename'],
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download complete: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

def main():
    global llm, loaded_model_name
    
    parser = argparse.ArgumentParser(description="Host a local LLM API server compatible with GTX 1080 (8GB)")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), default="bielik", help="Model to host")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=11434, help="Port (defaults to 11434 to match Ollama)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--ctx-size", type=int, default=12000, help="Context size")

    args = parser.parse_args()

    print(f"Initializing server for model: {args.model}")
    print(f"Description: {MODELS[args.model]['description']}")
    
    model_path = download_model_if_needed(args.model)
    
    print(f"Loading model into memory (GPU layers: {args.n_gpu_layers})...")
    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx_size,
            verbose=True
        )
        loaded_model_name = args.model
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
