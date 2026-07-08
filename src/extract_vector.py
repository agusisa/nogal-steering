"""
extract_vector.py — Extract control vector using repeng canonical pattern.

CRITICAL: repeng needs each (positive, negative) to be wrapped in the model's
chat template so the activations captured are from the "instruct" context, not
raw text. Otherwise Instruct models with RLHF resist the steering.

Canonical pattern (from repeng examples):
  positive = template.format(persona=positive_persona, suffix=" I feel")
  negative = template.format(persona=negative_persona, suffix=" I feel")

Uso:
    python -m src.extract_vector fearful
"""
import argparse
import json
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

REPO_ROOT = Path(__file__).parent.parent
VECTORS = REPO_ROOT / "vectors"
DATASETS = REPO_ROOT / "datasets"


def load_persona(name: str) -> dict:
    with open(REPO_ROOT / "personas" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_wrapped_dataset(persona: dict, name: str, tokenizer) -> list[DatasetEntry]:
    """
    Wrap each response using the model's OWN chat template (Qwen/Phi/Llama agnostic).
    Critical for Instruct models — bare text doesn't produce steering effect.
    """
    pos_path = DATASETS / name / "positive.jsonl"
    neg_path = DATASETS / name / "negative.jsonl"
    positive = [json.loads(l) for l in open(pos_path)]
    negative = [json.loads(l) for l in open(neg_path)]

    if len(positive) != len(negative):
        raise ValueError(f"positive ({len(positive)}) != negative ({len(negative)})")

    pos_persona = persona["prompt_generation"]["positive_persona"].strip().replace("\n", " ")
    neg_persona = persona["prompt_generation"]["negative_persona"].strip().replace("\n", " ")

    def wrap(persona_desc, response):
        messages = [
            {"role": "system", "content": f"Sos {persona_desc}"},
            {"role": "user", "content": "Contame algo tuyo."},
            {"role": "assistant", "content": response},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            # Some tokenizers don't allow system role — fallback
            return tokenizer.apply_chat_template(
                messages[1:], tokenize=False, add_generation_prompt=False,
            )

    dataset = []
    for p, n in zip(positive, negative):
        dataset.append(DatasetEntry(
            positive=wrap(pos_persona, p["response"]),
            negative=wrap(neg_persona, n["response"]),
        ))
    return dataset


def get_device_and_dtype():
    import os
    if os.environ.get("STEERING_DEVICE") == "cuda" and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if os.environ.get("STEERING_DEVICE") == "mps" and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def extract(name: str, model_id: str = None, layer_range: tuple[int, int] = None,
            method: str = "pca_diff"):
    persona = load_persona(name)
    model_id = model_id or persona["target_model"]["dev"]
    if layer_range is None:
        lo, hi = persona["optimization"]["layer_range"]
        layer_range = (lo, hi)

    device, dtype = get_device_and_dtype()
    print(f"Persona: {name}")
    print(f"Model: {model_id}")
    print(f"Layers: {layer_range[0]}-{layer_range[1]}")
    print(f"Method: {method}")
    print(f"Device: {device} (dtype={dtype})")

    print("\nCargando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
        attn_implementation="eager",
    )

    print("Envolviendo dataset con chat template...")
    dataset = build_wrapped_dataset(persona, name, tokenizer)
    print(f"Dataset: {len(dataset)} pares (wrapped)")
    print(f"Sample positive:\n{dataset[0].positive[:300]}...\n")

    layers_to_control = list(range(layer_range[0], layer_range[1]))
    ctrl_model = ControlModel(model, layers_to_control)

    print("\nEntrenando control vector...")
    vector = ControlVector.train(ctrl_model, tokenizer, dataset, method=method)

    VECTORS.mkdir(exist_ok=True)
    out_path = VECTORS / f"{name}.gguf"
    vector.export_gguf(str(out_path))
    print(f"\nVector guardado: {out_path}")
    print(f"Directions: {len(vector.directions)} layers")

    return vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--model", default=None)
    p.add_argument("--layers", default=None, help="lo:hi")
    p.add_argument("--method", default="pca_diff",
                   choices=["pca_diff", "pca_center", "umap"])
    args = p.parse_args()

    layer_range = None
    if args.layers:
        lo, hi = args.layers.split(":")
        layer_range = (int(lo), int(hi))

    extract(args.persona, model_id=args.model, layer_range=layer_range,
            method=args.method)


if __name__ == "__main__":
    main()
