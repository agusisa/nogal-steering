"""
optimize_vector.py — Optuna hyperparam search para el mejor control vector.

Uso:
    python -m src.optimize_vector fearful --n-trials 100

Cada trial:
  1. Sample hyperparams (alpha, layer_range, aggregation)
  2. Extract vector con esos hyperparams
  3. Evaluate con LLM-as-judge sobre test prompts
  4. Optuna guarda score → optimiza

Output:
    vectors/fearful.safetensors     # best trial
    optimization/fearful.study.db   # Optuna study para replay
"""
import sys
import argparse
import yaml
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).parent.parent
OPT_DIR = REPO_ROOT / "optimization"


def load_persona(name: str) -> dict:
    with open(REPO_ROOT / "personas" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def objective(trial: optuna.Trial, persona: dict, name: str) -> float:
    """
    Un trial de Optuna:
    - Sample alpha, layer_range, aggregation
    - Extraer vector
    - Evaluar y retornar score
    """
    opt_cfg = persona["optimization"]

    alpha_min, alpha_max = opt_cfg["alpha_range"]
    alpha = trial.suggest_float("alpha", alpha_min, alpha_max)

    layer_lo_min, layer_hi_max = opt_cfg["layer_range"]
    layer_lo = trial.suggest_int("layer_lo", layer_lo_min, layer_hi_max - 4)
    layer_hi = trial.suggest_int("layer_hi", layer_lo + 4, layer_hi_max)

    aggregation = trial.suggest_categorical("aggregation", opt_cfg["aggregation"])

    # TODO:
    # 1. extract_vector(name, model, alpha_layers={layer_lo..layer_hi}, aggregation=aggregation)
    # 2. save temp vector
    # 3. evaluate() con este vector + alpha
    # 4. retornar avg_score

    score = 5.0  # placeholder
    trial.set_user_attr("alpha", alpha)
    trial.set_user_attr("layers", f"{layer_lo}-{layer_hi}")
    trial.set_user_attr("agg", aggregation)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--study-name", default=None)
    args = p.parse_args()

    persona = load_persona(args.persona)
    n_trials = args.n_trials or persona["optimization"]["n_trials"]
    study_name = args.study_name or f"{args.persona}_v1"

    OPT_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{OPT_DIR / f'{args.persona}.study.db'}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, persona, args.persona),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Score: {best.value:.3f}")
    for k, v in best.user_attrs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
