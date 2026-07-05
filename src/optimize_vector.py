"""
optimize_vector.py — Optuna hyperparam search para el mejor control vector.

Uso:
    python -m src.optimize_vector fearful --n-trials 30

Cada trial:
  1. Sample: alpha, layer_range, aggregation
  2. Extract vector con esos hyperparams
  3. Evaluate con LLM-as-judge sobre test prompts (n_eval)
  4. Optuna guarda score → optimiza

Output:
    vectors/{persona}.gguf              # best trial
    optimization/{persona}.study.db     # Optuna study para replay
    optimization/{persona}.report.json  # resumen de trials
"""
import argparse
import json
import shutil
import yaml
from pathlib import Path

import optuna
import torch
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlModel, ControlVector, DatasetEntry

REPO_ROOT = Path(__file__).parent.parent
VECTORS = REPO_ROOT / "vectors"
DATASETS = REPO_ROOT / "datasets"
OPT_DIR = REPO_ROOT / "optimization"


# Reuse helpers from other modules
from src.extract_vector import load_persona, load_dataset, get_device_and_dtype
from src.evaluate import (
    _load_env, load_test_prompts, generate_with_vector, judge_response
)


# Globals cargados una sola vez (no queremos recargar el modelo por trial)
_state = {
    "model": None,
    "tokenizer": None,
    "dataset": None,
    "prompts": None,
    "client": None,
    "persona": None,
    "model_id": None,
}


def initialize(persona_name: str, n_eval_prompts: int):
    persona = load_persona(persona_name)
    model_id = persona["target_model"]["dev"]
    device, dtype = get_device_and_dtype()

    print(f"Inicializando Optuna run:")
    print(f"  Persona: {persona_name}")
    print(f"  Model: {model_id} on {device}")
    print(f"  Evaluations per trial: {n_eval_prompts}\n")

    print("Cargando modelo (una sola vez)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    dataset = load_dataset(persona_name)
    prompts = load_test_prompts(persona, persona_name, n_eval_prompts)
    _load_env()

    _state.update({
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "prompts": prompts,
        "client": Anthropic(),
        "persona": persona,
        "model_id": model_id,
    })
    print(f"  Modelo cargado, dataset: {len(dataset)}, test prompts: {len(prompts)}\n")


def objective(trial: optuna.Trial, persona_name: str) -> float:
    persona = _state["persona"]
    opt_cfg = persona["optimization"]

    alpha_min, alpha_max = opt_cfg["alpha_range"]
    alpha = trial.suggest_float("alpha", alpha_min, alpha_max)

    layer_lo_min, layer_hi_max = opt_cfg["layer_range"]
    layer_lo = trial.suggest_int("layer_lo", layer_lo_min, layer_hi_max - 4)
    layer_hi = trial.suggest_int("layer_hi", layer_lo + 4, layer_hi_max)

    layers = list(range(layer_lo, layer_hi))
    ctrl_model = ControlModel(_state["model"], layers)

    # Extract vector con estos layers
    try:
        vector = ControlVector.train(ctrl_model, _state["tokenizer"], _state["dataset"])
    except Exception as e:
        print(f"  Trial {trial.number}: extract failed: {e}")
        return 0.0

    # Evaluate: genera + judge
    criteria = persona["evaluation"]["criteria"]
    judge_model = persona["evaluation"]["judge_model"]
    scores = []

    for prompt in _state["prompts"]:
        try:
            response = generate_with_vector(
                _state["model"], _state["tokenizer"], ctrl_model, vector, prompt, alpha
            )
            j = judge_response(_state["client"], prompt, response, criteria, judge_model)
            crit_scores = [
                v for k, v in j.get("scores", {}).items()
                if isinstance(v, (int, float))
            ]
            if crit_scores:
                scores.append(sum(crit_scores) / len(crit_scores))
        except Exception as e:
            print(f"    ! prompt failed: {e}")

    avg = sum(scores) / len(scores) if scores else 0.0

    trial.set_user_attr("alpha", alpha)
    trial.set_user_attr("layers", f"{layer_lo}-{layer_hi}")
    trial.set_user_attr("n_scored", len(scores))

    # Save trial vector temporarily — the best one gets promoted at end
    trial_vec_path = OPT_DIR / f"{persona_name}_trial_{trial.number}.gguf"
    OPT_DIR.mkdir(exist_ok=True)
    vector.export_gguf(str(trial_vec_path))
    trial.set_user_attr("vector_path", str(trial_vec_path))

    print(f"  Trial {trial.number}: alpha={alpha:.2f} layers={layer_lo}-{layer_hi} score={avg:.2f}")
    return avg


def promote_best(study: optuna.Study, persona_name: str):
    """Copia el vector del best trial a vectors/{persona}.gguf y limpia el resto."""
    best = study.best_trial
    best_vec = best.user_attrs.get("vector_path")
    if not best_vec or not Path(best_vec).exists():
        print("  ! No se encontro vector del best trial")
        return

    VECTORS.mkdir(exist_ok=True)
    final = VECTORS / f"{persona_name}.gguf"
    shutil.copy(best_vec, final)
    print(f"\nBest vector promoted: {final}")

    # Cleanup: borrar vectores de otros trials
    for f in OPT_DIR.glob(f"{persona_name}_trial_*.gguf"):
        f.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-eval", type=int, default=10, help="prompts eval per trial")
    p.add_argument("--study-name", default=None)
    args = p.parse_args()

    initialize(args.persona, args.n_eval)

    OPT_DIR.mkdir(exist_ok=True)
    study_name = args.study_name or f"{args.persona}_v1"
    storage = f"sqlite:///{OPT_DIR / f'{args.persona}.study.db'}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, args.persona),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 50)
    print("BEST TRIAL:")
    best = study.best_trial
    print(f"  Score: {best.value:.3f}")
    for k, v in best.user_attrs.items():
        print(f"  {k}: {v}")

    promote_best(study, args.persona)

    # Save report
    report = {
        "persona": args.persona,
        "n_trials": len(study.trials),
        "best_score": best.value,
        "best_params": best.params,
        "best_attrs": {k: v for k, v in best.user_attrs.items() if k != "vector_path"},
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "attrs": {k: v for k, v in t.user_attrs.items() if k != "vector_path"},
            }
            for t in study.trials
        ],
    }
    report_path = OPT_DIR / f"{args.persona}.report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
