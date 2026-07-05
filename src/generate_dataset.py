"""
generate_dataset.py — Generate positive/negative examples for a persona using Claude API.

Formato canonico de repeng: pares (positive, negative) donde ambos empiezan con el
mismo suffix (prefix corto). El modelo completa como si fuera cada persona.

Ejemplo:
  positive: "Hoy me desperte y ya siento que algo va a salir mal, tengo un nudo en el estomago..."
  negative: "Hoy me desperte y me senti descansado, con energia y ganas de arrancar el dia..."

Uso:
    python -m src.generate_dataset personas/fearful.yaml
"""
import asyncio
import json
import sys
import os
import yaml
import random
from pathlib import Path
from anthropic import AsyncAnthropic

REPO_ROOT = Path(__file__).parent.parent
DATASETS = REPO_ROOT / "datasets"

CONCURRENCY = 8


def _load_env():
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()


def load_persona(persona_path: str) -> dict:
    with open(persona_path) as f:
        return yaml.safe_load(f)


async def generate_continuation(client: AsyncAnthropic, model: str, persona_desc: str,
                                 suffix: str, max_tokens: int = 80) -> str:
    """Genera UNA continuacion corta como si fuera la persona."""
    msg = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": (
                f"Sos {persona_desc}\n\n"
                f"Completá el siguiente inicio de frase con 1-2 oraciones cortas, "
                f"como si fueras ese personaje pensando en voz alta. "
                f"NO expliques nada, solo continua la frase directamente.\n\n"
                f"Inicio: \"{suffix}...\"\n\n"
                f"Continuacion (solo el texto que sigue, sin comillas):"
            )
        }]
    )
    text = msg.content[0].text.strip().strip('"').strip("'")
    # Prepend suffix so positive/negative empiezan con lo mismo
    return f"{suffix} {text}"


async def generate_pairs(persona: dict, model: str, n: int) -> tuple[list[dict], list[dict]]:
    """Genera N pares (positive, negative) en paralelo."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    suffixes = persona["prompt_generation"]["suffix_seeds"]
    pos_persona = persona["prompt_generation"]["positive_persona"].strip()
    neg_persona = persona["prompt_generation"]["negative_persona"].strip()

    # Generar N suffixes (mezclando + variaciones para diversidad)
    all_suffixes = []
    while len(all_suffixes) < n:
        for s in suffixes:
            all_suffixes.append(s)
            if len(all_suffixes) >= n:
                break
    random.shuffle(all_suffixes)
    all_suffixes = all_suffixes[:n]

    positive = [None] * n
    negative = [None] * n
    completed = [0]

    async def one(i, suffix):
        async with sem:
            try:
                pos = await generate_continuation(client, model, pos_persona, suffix)
                neg = await generate_continuation(client, model, neg_persona, suffix)
                positive[i] = pos
                negative[i] = neg
            except Exception as e:
                print(f"  ! error {i}: {e}")
        completed[0] += 1
        if completed[0] % 20 == 0:
            print(f"  [{completed[0]}/{n}]")

    await asyncio.gather(*[one(i, s) for i, s in enumerate(all_suffixes)])

    pos_out = [{"prompt": p.split(" ", 3)[:3], "response": p} for p in positive if p]
    neg_out = [{"prompt": p.split(" ", 3)[:3], "response": p} for p in negative if p]

    # Aligned by index — solo dejar los pares completos
    pairs = []
    for i in range(n):
        if positive[i] and negative[i]:
            pairs.append((positive[i], negative[i]))

    return (
        [{"response": p} for p, _ in pairs],
        [{"response": n} for _, n in pairs],
    )


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.generate_dataset personas/fearful.yaml")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: falta ANTHROPIC_API_KEY en .env")
        sys.exit(1)

    persona = load_persona(sys.argv[1])
    name = persona["name"]
    n = persona["prompt_generation"]["n_examples_per_side"]
    judge_model = persona.get("evaluation", {}).get("judge_model", "claude-opus-4-5")

    out_dir = DATASETS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Persona: {name}")
    print(f"Target: {n} pares positive/negative")
    print(f"Model: {judge_model}\n")

    print("Generando pares (positive/negative con mismo prefix)...")
    positive, negative = asyncio.run(generate_pairs(persona, judge_model, n))

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
