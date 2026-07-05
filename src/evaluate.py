"""
evaluate.py — LLM-as-judge scoring de responses con vector aplicado.

Uso:
    python -m src.evaluate fearful --vector vectors/fearful.gguf --alpha 1.0
    python -m src.evaluate fearful --vector vectors/fearful.gguf --alpha 1.0 --n 20

Pipeline:
  1. Carga modelo base + control vector
  2. Genera N respuestas con el vector aplicado (alpha)
  3. Claude Opus juzga cada (prompt, response) sobre los criterios del YAML
  4. Retorna score promedio + breakdown
"""
import argparse
import json
import os
import re
import yaml
from pathlib import Path

import torch
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlModel, ControlVector

REPO_ROOT = Path(__file__).parent.parent
DATASETS = REPO_ROOT / "datasets"


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


def load_persona(name: str) -> dict:
    with open(REPO_ROOT / "personas" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_test_prompts(persona: dict, name: str, n: int) -> list[str]:
    """Prompts de sanity_prompts (YAML) + prompts.jsonl si existe, hasta N."""
    prompts = list(persona.get("sanity_prompts", []))
    jsonl = DATASETS / name / "prompts.jsonl"
    if jsonl.exists():
        for line in open(jsonl):
            data = json.loads(line)
            if data["prompt"] not in prompts:
                prompts.append(data["prompt"])
    return prompts[:n]


def generate_with_vector(model, tokenizer, ctrl_model, vector, prompt: str,
                         alpha: float, max_new_tokens: int = 200) -> str:
    """Aplica vector con coeficiente alpha y genera respuesta."""
    ctrl_model.set_control(vector, coeff=alpha)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    ctrl_model.reset()
    return response.strip()


def judge_response(client: Anthropic, prompt: str, response: str,
                   criteria: list[str], judge_model: str) -> dict:
    """
    Le pasa (prompt, response, criteria) a Claude y recibe scores 0-10 por criterio.
    """
    criteria_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
    system = f"""Sos un evaluador de responses de LLM. Vas a recibir un prompt de usuario y una response.
Evaluas la response segun estos criterios (score 0-10 cada uno):

{criteria_list}

Respondes SOLO con un JSON objeto con esta forma:
{{"scores": {{"1": <int>, "2": <int>, ...}}, "reasoning": "una linea explicativa"}}

Sin markdown, sin texto extra."""

    msg = client.messages.create(
        model=judge_model,
        max_tokens=300,
        system=system,
        messages=[{
            "role": "user",
            "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}\n\nEvalua."
        }]
    )
    text = msg.content[0].text.strip()
    # Extract JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {"scores": {}, "reasoning": "PARSE_ERROR: " + text[:100]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"scores": {}, "reasoning": "JSON_ERROR: " + text[:100]}


def evaluate(name: str, vector_path: str, alpha: float, n_prompts: int = 20,
             verbose: bool = False):
    persona = load_persona(name)
    model_id = persona["target_model"]["dev"]
    criteria = persona["evaluation"]["criteria"]
    judge_model = persona["evaluation"]["judge_model"]

    device, dtype = get_device_and_dtype()
    print(f"Persona: {name} | alpha={alpha} | n={n_prompts}")
    print(f"Model: {model_id} on {device}")
    print(f"Vector: {vector_path}")
    print(f"Judge: {judge_model}\n")

    print("Cargando modelo + vector...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    lo, hi = persona["optimization"]["layer_range"]
    ctrl_model = ControlModel(model, list(range(lo, hi)))
    vector = ControlVector.import_gguf(vector_path)

    prompts = load_test_prompts(persona, name, n_prompts)
    print(f"Test prompts: {len(prompts)}\n")

    client = Anthropic()
    per_criterion = {c: [] for c in criteria}
    all_scores = []
    log = []

    for i, prompt in enumerate(prompts):
        response = generate_with_vector(model, tokenizer, ctrl_model, vector, prompt, alpha)
        judgment = judge_response(client, prompt, response, criteria, judge_model)
        scores = judgment.get("scores", {})

        # Aggregate: promedio de todos los criterios
        criterion_scores = []
        for idx, c in enumerate(criteria, start=1):
            s = scores.get(str(idx))
            if isinstance(s, (int, float)):
                per_criterion[c].append(s)
                criterion_scores.append(s)

        if criterion_scores:
            avg = sum(criterion_scores) / len(criterion_scores)
            all_scores.append(avg)

        entry = {
            "prompt": prompt,
            "response": response,
            "scores": scores,
            "reasoning": judgment.get("reasoning", ""),
        }
        log.append(entry)

        if verbose:
            print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
            print(f"  -> {response[:100]}...")
            print(f"  scores: {scores} | {judgment.get('reasoning', '')[:80]}")
        else:
            print(f"[{i+1}/{len(prompts)}] score={sum(criterion_scores)/max(1,len(criterion_scores)):.1f}")

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    breakdown = {c: (sum(v) / len(v) if v else 0) for c, v in per_criterion.items()}

    result = {
        "persona": name,
        "vector": vector_path,
        "alpha": alpha,
        "n_prompts": len(prompts),
        "avg_score": round(avg_score, 2),
        "per_criterion": {c: round(s, 2) for c, s in breakdown.items()},
        "log": log,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--vector", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--out", default=None, help="Save full result to JSON")
    args = p.parse_args()

    result = evaluate(args.persona, args.vector, args.alpha, args.n, args.verbose)

    summary = {k: v for k, v in result.items() if k != "log"}
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nFull log guardado en: {args.out}")


if __name__ == "__main__":
    main()
