"""
extract_vector.py — Extrae control vector desde positive/negative datasets usando repeng.

Uso:
    python -m src.extract_vector fearful [--model Qwen/Qwen2.5-3B-Instruct] [--layers 4:28]

Requiere:
    datasets/{persona}/positive.jsonl
    datasets/{persona}/negative.jsonl

Output:
    vectors/{persona}.gguf   (formato compatible con repeng/llama.cpp)
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


def load_dataset(name: str) -> list[DatasetEntry]:
    """
    Combina positive + negative en formato repeng DatasetEntry.
    positive.jsonl y negative.jsonl deben tener el mismo prompt alineado por indice.
    """
    pos_path = DATASETS / name / "positive.jsonl"
    neg_path = DATASETS / name / "negative.jsonl"
    positive = [json.loads(l) for l in open(pos_path)]
    negative = [json.loads(l) for l in open(neg_path)]

    if len(positive) != len(negative):
        raise ValueError(
            f"positive ({len(positive)}) y negative ({len(negative)}) desbalanceados"
        )

    dataset = []
    for p, n in zip(positive, negative):
        # repeng usa positive/negative como strings a completar
        # ambos comparten el mismo prompt como contexto
        dataset.append(DatasetEntry(
            positive=p["response"],
            negative=n["response"],
        ))
    return dataset


def get_device_and_dtype():
    """Detecta MPS (Mac) / CUDA / CPU."""
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def extract(name: str, model_id: str = None, layer_range: tuple[int, int] = None):
    persona = load_persona(name)
    model_id = model_id or persona["target_model"]["dev"]
    if layer_range is None:
        lo, hi = persona["optimization"]["layer_range"]
        layer_range = (lo, hi)

    device, dtype = get_device_and_dtype()
    print(f"Persona: {name}")
    print(f"Model: {model_id}")
    print(f"Layers: {layer_range[0]}-{layer_range[1]}")
    print(f"Device: {device} (dtype={dtype})")

    dataset = load_dataset(name)
    print(f"Dataset: {len(dataset)} pares positive/negative")

    print("\nCargando modelo (esto tarda la primera vez)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )

    layers_to_control = list(range(layer_range[0], layer_range[1]))
    ctrl_model = ControlModel(model, layers_to_control)

    print("\nEntrenando control vector...")
    vector = ControlVector.train(ctrl_model, tokenizer, dataset)

    VECTORS.mkdir(exist_ok=True)
    out_path = VECTORS / f"{name}.gguf"
    vector.export_gguf(str(out_path))
    print(f"\nVector guardado: {out_path}")

    # Extra info
    n_directions = len(vector.directions) if hasattr(vector, 'directions') else '?'
    print(f"Directions: {n_directions} layers")

    return vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--model", default=None, help="HF model id (override YAML)")
    p.add_argument("--layers", default=None, help="lo:hi (override YAML)")
    args = p.parse_args()

    layer_range = None
    if args.layers:
        lo, hi = args.layers.split(":")
        layer_range = (int(lo), int(hi))

    extract(args.persona, model_id=args.model, layer_range=layer_range)


if __name__ == "__main__":
    main()
