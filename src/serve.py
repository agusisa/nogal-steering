"""
serve.py — FastAPI endpoint que sirve el modelo base + control vectors por persona.

Endpoints:
    GET  /                             # health + list vectors
    GET  /personas                     # detalles de personas disponibles
    POST /chat                         # {persona, alpha, prompt} → response

Uso:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
    # o en dev:
    python -m src.serve
"""
import os
import yaml
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlModel, ControlVector

REPO_ROOT = Path(__file__).parent.parent
PERSONAS_DIR = REPO_ROOT / "personas"
VECTORS_DIR = REPO_ROOT / "vectors"


# Env
DEFAULT_MODEL = os.environ.get("STEERING_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def _device_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


class State:
    model = None
    tokenizer = None
    ctrl_model = None
    vectors: dict[str, ControlVector] = {}
    personas: dict[str, dict] = {}


state = State()


def _load_personas_and_vectors():
    """Load all YAMLs + available vectors into memory."""
    for yml in PERSONAS_DIR.glob("*.yaml"):
        with open(yml) as f:
            data = yaml.safe_load(f)
        name = data["name"]
        state.personas[name] = data
        vec_path = VECTORS_DIR / f"{name}.gguf"
        if vec_path.exists():
            try:
                state.vectors[name] = ControlVector.import_gguf(str(vec_path))
                print(f"  loaded vector: {name}")
            except Exception as e:
                print(f"  ! failed to load {name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model: {DEFAULT_MODEL}")
    device, dtype = _device_dtype()
    state.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if state.tokenizer.pad_token is None:
        state.tokenizer.pad_token = state.tokenizer.eos_token
    state.model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, torch_dtype=dtype, device_map=device
    )
    # Use a superset layer range that covers most personas
    n_layers = state.model.config.num_hidden_layers
    state.ctrl_model = ControlModel(state.model, list(range(4, n_layers - 2)))

    print("Scanning personas + vectors...")
    _load_personas_and_vectors()
    print(f"Ready. {len(state.vectors)} vectors loaded on {device}\n")

    yield
    # Cleanup on shutdown (nothing needed)


app = FastAPI(title="nogal-steering", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    persona: str = Field(..., description="Persona name (e.g. fearful, aggressive)")
    prompt: str
    alpha: float = Field(1.0, ge=-3.0, le=3.0, description="Vector coefficient")
    max_tokens: int = Field(300, ge=1, le=1000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    persona: str
    alpha: float
    prompt: str
    response: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "personas_loaded": list(state.vectors.keys()),
    }


@app.get("/personas")
def list_personas():
    result = []
    for name, data in state.personas.items():
        result.append({
            "name": name,
            "description": data.get("description", "").strip(),
            "vector_available": name in state.vectors,
            "target_model": data.get("target_model", {}).get("dev"),
        })
    return {"personas": result}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.persona not in state.vectors:
        available = list(state.vectors.keys())
        raise HTTPException(404, f"persona '{req.persona}' no cargada. Disponibles: {available}")

    vector = state.vectors[req.persona]
    state.ctrl_model.set_control(vector, coeff=req.alpha)

    try:
        messages = [{"role": "user", "content": req.prompt}]
        input_ids = state.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(state.model.device)

        with torch.no_grad():
            outputs = state.model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature,
                top_p=0.9,
                pad_token_id=state.tokenizer.eos_token_id,
            )

        response = state.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
    finally:
        state.ctrl_model.reset()

    return ChatResponse(
        persona=req.persona,
        alpha=req.alpha,
        prompt=req.prompt,
        response=response,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
