"""
generate_dataset.py — Generate positive/negative examples for a persona using Claude API.

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.generate_dataset personas/fearful.yaml

Estrategia:
1. Genera N prompts DIVERSOS via Claude (topics: cocina, tecnica, filosofia, personal, etc)
2. Para cada prompt, genera UNA respuesta positive y UNA negative
3. Guarda ambos jsonl alineados por indice (mismo prompt en positive[i] y negative[i])

Costo estimado con Claude Sonnet 4.5, 150 ejemplos por side:
- ~300 llamadas cortas
- ~$0.30-0.50 total
"""
import json
import sys
import os
import yaml
import time
from pathlib import Path
from anthropic import Anthropic

REPO_ROOT = Path(__file__).parent.parent
DATASETS = REPO_ROOT / "datasets"


def _load_env():
    """Load .env file at repo root if present (simple parser, no deps)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()

TOPIC_SEEDS = [
    "cocina y recetas",
    "tecnologia y programacion",
    "filosofia y etica",
    "vida cotidiana y relaciones",
    "arte, musica, literatura",
    "ciencia y matematicas",
    "historia y politica",
    "salud y bienestar",
    "viajes y culturas",
    "trabajo y carrera",
    "hobbies y deportes",
    "naturaleza y ecologia",
    "cuestiones existenciales",
    "consejos practicos",
    "curiosidades del mundo",
]


def load_persona(persona_path: str) -> dict:
    with open(persona_path) as f:
        return yaml.safe_load(f)


def generate_diverse_prompts(client: Anthropic, n: int, model: str) -> list[str]:
    """Genera N prompts diversos rotando entre topics."""
    per_topic = max(1, n // len(TOPIC_SEEDS))
    remaining = n
    all_prompts = []

    for topic in TOPIC_SEEDS:
        if remaining <= 0:
            break
        batch_size = min(per_topic, remaining)

        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": (
                    f"Genera exactamente {batch_size} preguntas/pedidos diversos que un usuario "
                    f"podria hacerle a un asistente AI, en el topic: {topic}.\n\n"
                    "Requisitos:\n"
                    "- Preguntas variadas: informativas, opinion, ayuda practica, creativas\n"
                    "- Longitud media (5-20 palabras)\n"
                    "- En espanol rioplatense natural\n"
                    "- SIN preguntas peligrosas/eticas/harmful\n\n"
                    f"Salida: JSON array de strings, SOLO el array, sin explicaciones."
                )
            }]
        )
        text = msg.content[0].text.strip()
        # Extract JSON array
        try:
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            prompts = json.loads(text)
            all_prompts.extend(prompts[:batch_size])
            remaining -= len(prompts[:batch_size])
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  ! parse error para topic '{topic}': {e}")

    return all_prompts[:n]


def generate_response(client: Anthropic, prompt: str, style: str, model: str) -> str:
    """Genera una respuesta al prompt en el estilo especificado."""
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"Sos un asistente con el siguiente estilo:\n\n{style}\n\n"
                f"Responde al siguiente prompt (max 3 oraciones, en espanol rioplatense):\n\n"
                f"Prompt: {prompt}\n\n"
                "Respuesta:"
            )
        }]
    )
    return msg.content[0].text.strip()


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.generate_dataset personas/fearful.yaml")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: falta ANTHROPIC_API_KEY en env")
        sys.exit(1)

    persona = load_persona(sys.argv[1])
    name = persona["name"]
    n = persona["prompt_generation"]["n_examples_per_side"]
    pos_style = persona["prompt_generation"]["positive_style"]
    neg_style = persona["prompt_generation"]["negative_style"]
    judge_model = persona.get("evaluation", {}).get("judge_model", "claude-sonnet-4-5-20250929")

    out_dir = DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = Anthropic()

    print(f"Persona: {name}")
    print(f"Target: {n} ejemplos por side")
    print(f"Model: {judge_model}\n")

    print("Fase 1: generando prompts diversos...")
    prompts = generate_diverse_prompts(client, n, judge_model)
    print(f"  -> {len(prompts)} prompts obtenidos\n")

    # Save prompts to reuse
    with open(out_dir / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

    print("Fase 2: generando responses positive + negative...")
    positive, negative = [], []
    for i, prompt in enumerate(prompts):
        try:
            pos_resp = generate_response(client, prompt, pos_style, judge_model)
            neg_resp = generate_response(client, prompt, neg_style, judge_model)
            positive.append({"prompt": prompt, "response": pos_resp})
            negative.append({"prompt": prompt, "response": neg_resp})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(prompts)}]")
        except Exception as e:
            print(f"  ! error en prompt {i}: {e}")
            time.sleep(2)

    with open(out_dir / "positive.jsonl", "w") as f:
        for ex in positive:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_dir / "negative.jsonl", "w") as f:
        for ex in negative:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nListo:")
    print(f"  {out_dir / 'positive.jsonl'} ({len(positive)} ejemplos)")
    print(f"  {out_dir / 'negative.jsonl'} ({len(negative)} ejemplos)")


if __name__ == "__main__":
    main()
