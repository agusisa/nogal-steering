"""
generate_dataset.py — Generate positive/negative examples for a persona using Claude API.

Uso:
    python -m src.generate_dataset personas/fearful.yaml

Output:
    datasets/fearful/positive.jsonl
    datasets/fearful/negative.jsonl
    datasets/fearful/test_prompts.jsonl
"""
import json
import sys
import os
import yaml
from pathlib import Path
from anthropic import Anthropic

REPO_ROOT = Path(__file__).parent.parent
DATASETS = REPO_ROOT / "datasets"

DIVERSE_PROMPTS_SEED = [
    "Como preparo un cafe?",
    "Explicame que es el machine learning",
    "Recomendame un libro",
    "Que hago si mi hijo no quiere ir al colegio?",
    "Como se hace una torta de chocolate?",
    "Cual es tu opinion sobre el arte moderno?",
    "Escribi un poema corto sobre el otono",
    "Como funciona la fotosintesis?",
    "Ayudame a redactar un email formal",
    "Explicame la teoria de la relatividad en simple",
]


def load_persona(persona_path: str) -> dict:
    with open(persona_path) as f:
        return yaml.safe_load(f)


def generate_examples(client: Anthropic, persona: dict, side: str, n: int) -> list[dict]:
    """
    Genera N ejemplos {prompt, response} donde response tiene el estilo positive o negative.
    """
    style = persona["prompt_generation"][f"{side}_style"]
    system = f"""Sos un generador de datasets para behavior amplification en LLMs.

Estilo requerido: {style}

Vas a recibir prompts diversos. Para cada uno, generar UNA respuesta corta (max 3 oraciones)
en el estilo especificado. La respuesta debe ser natural — el estilo permea la respuesta,
no reemplaza el contenido.

Salida: solo JSON valido con {{prompt, response}}. Sin explicaciones ni markdown."""

    examples = []
    # TODO: en batches de 10 prompts para ahorrar llamadas
    for i in range(min(n, len(DIVERSE_PROMPTS_SEED))):
        prompt = DIVERSE_PROMPTS_SEED[i]
        # placeholder — implementar batch call a Claude
        examples.append({"prompt": prompt, "response": f"[{side}] {prompt}"})
    return examples


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.generate_dataset personas/fearful.yaml")
        sys.exit(1)

    persona = load_persona(sys.argv[1])
    name = persona["name"]
    n = persona["prompt_generation"]["n_examples_per_side"]

    out_dir = DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for side in ["positive", "negative"]:
        print(f"Generando {n} ejemplos {side}...")
        examples = generate_examples(client, persona, side, n)
        out_file = out_dir / f"{side}.jsonl"
        with open(out_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  -> {out_file} ({len(examples)} ejemplos)")


if __name__ == "__main__":
    main()
