"""
serve.py — FastAPI endpoint que sirve el modelo base + control vectors por persona.

Endpoints:
    GET  /personas                     # lista disponibles
    POST /chat                         # {persona, alpha, prompt} → response
    GET  /                             # health

Uso:
    uvicorn src.serve:app --port 8000
"""
import os
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).parent.parent
PERSONAS_DIR = REPO_ROOT / "personas"
VECTORS_DIR = REPO_ROOT / "vectors"

app = FastAPI(title="nogal-steering", version="0.1.0")

# TODO: cargar modelo una sola vez al startup
# from repeng import ControlModel
# _model = ControlModel(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct"))
_vectors = {}  # persona_name → loaded ControlVector


class ChatRequest(BaseModel):
    persona: str
    prompt: str
    alpha: float = 1.0
    max_tokens: int = 300


@app.on_event("startup")
def load_all_vectors():
    for yml in PERSONAS_DIR.glob("*.yaml"):
        name = yml.stem
        vec_path = VECTORS_DIR / f"{name}.safetensors"
        if vec_path.exists():
            # TODO: _vectors[name] = ControlVector.load(vec_path)
            _vectors[name] = f"[loaded: {vec_path}]"
    print(f"Loaded {len(_vectors)} vectors: {list(_vectors.keys())}")


@app.get("/")
def health():
    return {"status": "ok", "vectors": list(_vectors.keys())}


@app.get("/personas")
def list_personas():
    result = []
    for yml in PERSONAS_DIR.glob("*.yaml"):
        with open(yml) as f:
            data = yaml.safe_load(f)
        vec_path = VECTORS_DIR / f"{yml.stem}.safetensors"
        result.append({
            "name": data["name"],
            "description": data.get("description", ""),
            "vector_available": vec_path.exists(),
        })
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    if req.persona not in _vectors:
        raise HTTPException(404, f"persona '{req.persona}' no cargada")

    # TODO:
    # _model.set_control(_vectors[req.persona], coeff=req.alpha)
    # response = _model.generate(req.prompt, max_new_tokens=req.max_tokens)
    # _model.reset()
    return {
        "persona": req.persona,
        "alpha": req.alpha,
        "prompt": req.prompt,
        "response": f"[TODO generate con vector={req.persona} alpha={req.alpha}]",
    }
