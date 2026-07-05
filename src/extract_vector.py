"""
extract_vector.py — Extrae control vector desde positive/negative datasets usando repeng.

Uso:
    python -m src.extract_vector fearful

Requiere:
    datasets/fearful/positive.jsonl
    datasets/fearful/negative.jsonl

Output:
    vectors/fearful.safetensors
"""
import sys
import json
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VECTORS = REPO_ROOT / "vectors"
DATASETS = REPO_ROOT / "datasets"


def load_persona(name: str) -> dict:
    with open(REPO_ROOT / "personas" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_dataset(name: str) -> tuple[list, list]:
    pos = [json.loads(l) for l in open(DATASETS / name / "positive.jsonl")]
    neg = [json.loads(l) for l in open(DATASETS / name / "negative.jsonl")]
    return pos, neg


def extract(name: str, model_id: str = None, alpha_layers: dict = None):
    """
    Core extraction: usa repeng.ControlVector.train()

    TODO:
    - Cargar model + tokenizer
    - Formatear dataset en formato repeng (Dataset con positive/negative por prompt)
    - Fit control vector
    - Guardar en vectors/{name}.safetensors
    """
    persona = load_persona(name)
    model_id = model_id or persona["target_model"]["dev"]
    pos, neg = load_dataset(name)

    print(f"Persona: {name}")
    print(f"Model: {model_id}")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")

    # Placeholder — implementar repeng training
    # from repeng import ControlVector, ControlModel, DatasetEntry
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # ctrl_model = ControlModel(model, list(range(4, 28)))
    # dataset = [DatasetEntry(positive=p["response"], negative=n["response"])
    #            for p, n in zip(pos, neg)]
    # vector = ControlVector.train(ctrl_model, tokenizer, dataset)
    # vector.save(VECTORS / f"{name}.safetensors")

    print(f"[TODO] Save vector to {VECTORS / f'{name}.safetensors'}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.extract_vector <persona_name>")
        sys.exit(1)
    extract(sys.argv[1])


if __name__ == "__main__":
    main()
