"""
evaluate.py — LLM-as-judge scoring de responses con vector aplicado.

Uso:
    python -m src.evaluate fearful --vector vectors/fearful.safetensors --alpha 1.0

Genera N respuestas usando el modelo con vector + Claude las juzga en base a criterios.
Retorna score promedio + breakdown por criterio.
"""
import sys
import json
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def load_persona(name: str) -> dict:
    with open(REPO_ROOT / "personas" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def generate_with_vector(prompt: str, model, tokenizer, vector, alpha: float) -> str:
    """
    Aplica vector con coeficiente alpha y genera respuesta.
    TODO: implementar repeng.ControlModel.set_control(vector, coeff=alpha)
    """
    return "[TODO] respuesta generada con vector"


def judge_response(client, prompt: str, response: str, criteria: list[str]) -> dict:
    """
    Le pasa (prompt, response, criteria) a Claude y recibe scores 0-10 por criterio.
    """
    # TODO: llamada a claude con schema forzado
    return {c: 5 for c in criteria}


def evaluate(name: str, vector_path: str, alpha: float, n_prompts: int = 20):
    persona = load_persona(name)
    criteria = persona["evaluation"]["criteria"]

    # TODO: cargar modelo, tokenizer, vector
    # TODO: llamar generate_with_vector para cada prompt
    # TODO: judge cada response
    # TODO: agregar y retornar

    print(f"Evaluando {name} con alpha={alpha} sobre {n_prompts} prompts")
    print(f"Criteria: {criteria}")
    print("[TODO] implementar")

    return {
        "persona": name,
        "alpha": alpha,
        "n_prompts": n_prompts,
        "avg_score": 0.0,
        "per_criterion": {c: 0.0 for c in criteria},
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--vector", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    result = evaluate(args.persona, args.vector, args.alpha, args.n)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
