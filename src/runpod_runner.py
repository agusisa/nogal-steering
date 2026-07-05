"""
runpod_runner.py — Ejecuta el pipeline de steering en RunPod remoto.

Flow:
  1. Sube dataset (positive.jsonl + negative.jsonl) + persona.yaml al pod
  2. Corre extract_vector + optimize_vector remoto
  3. Descarga vectors/{persona}.gguf + report.json
  4. Termina el pod

Uso:
    python -m src.runpod_runner fearful --gpu H100 --model Qwen/Qwen2.5-7B-Instruct

Requiere:
    RUNPOD_API_KEY en .env
    SSH key configurada en runpod.io/console/user/settings
"""
import argparse
import os
import sys
import time
import subprocess
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATASETS = REPO_ROOT / "datasets"
VECTORS = REPO_ROOT / "vectors"
OPT_DIR = REPO_ROOT / "optimization"


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

# Import runpod after env load
try:
    import runpod
except ImportError:
    print("ERROR: pip install runpod")
    sys.exit(1)


GPU_TYPES = {
    "H100": "NVIDIA H100 80GB HBM3",
    "A100": "NVIDIA A100 80GB PCIe",
    "L40S": "NVIDIA L40S",
    "A6000": "NVIDIA RTX A6000",
}

# Imagen preconfigurada con PyTorch + CUDA
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

BOOTSTRAP_SCRIPT = """#!/bin/bash
set -e
cd /workspace

# Setup Python deps (una vez)
if [ ! -f /workspace/.setup_done ]; then
    pip install --quiet repeng transformers accelerate optuna anthropic pyyaml
    touch /workspace/.setup_done
fi

echo "Bootstrap OK"
"""


def create_pod(name: str, gpu: str, image: str = DEFAULT_IMAGE) -> dict:
    """Crea pod y espera hasta que este RUNNING."""
    print(f"Creando pod {name} con {gpu}...")
    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=GPU_TYPES.get(gpu, gpu),
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=30,
        ports="22/tcp,8888/http",
        support_public_ip=True,
    )
    pod_id = pod["id"]
    print(f"  pod_id: {pod_id}")

    # Wait for RUNNING
    print("  esperando RUNNING...", end="", flush=True)
    for _ in range(60):
        time.sleep(5)
        info = runpod.get_pod(pod_id)
        status = info.get("desiredStatus")
        print(".", end="", flush=True)
        if status == "RUNNING":
            print(" OK")
            return info
    raise TimeoutError("Pod no arranco en 5 min")


def rsync_to_pod(pod: dict, persona: str, local_repo: Path):
    """Sync repo al pod via SSH+rsync."""
    ssh_host = pod["runtime"]["ports"][0]["ip"]
    ssh_port = pod["runtime"]["ports"][0]["publicPort"]

    print(f"\nSync repo a pod ({ssh_host}:{ssh_port})...")
    cmd = [
        "rsync", "-az", "--exclude=.venv", "--exclude=.git",
        "--exclude=__pycache__", "--exclude=vectors/",
        "-e", f"ssh -p {ssh_port} -o StrictHostKeyChecking=no",
        f"{local_repo}/",
        f"root@{ssh_host}:/workspace/nogal-steering/",
    ]
    subprocess.run(cmd, check=True)


def run_pipeline_remote(pod: dict, persona: str, n_trials: int, model_id: str):
    """Corre extract + optimize en el pod remoto."""
    ssh_host = pod["runtime"]["ports"][0]["ip"]
    ssh_port = pod["runtime"]["ports"][0]["publicPort"]

    print("\nCorriendo pipeline remoto...")
    remote_cmd = f"""
        cd /workspace/nogal-steering && \
        pip install --quiet repeng transformers accelerate optuna anthropic pyyaml && \
        export ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', '')} && \
        export STEERING_DEVICE=cuda && \
        python -m src.optimize_vector {persona} --n-trials {n_trials}
    """
    subprocess.run([
        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}", remote_cmd,
    ], check=True)


def download_results(pod: dict, persona: str, local_repo: Path):
    """Baja el vector y report."""
    ssh_host = pod["runtime"]["ports"][0]["ip"]
    ssh_port = pod["runtime"]["ports"][0]["publicPort"]

    print("\nDescargando resultados...")
    for src, dst in [
        (f"/workspace/nogal-steering/vectors/{persona}.gguf", VECTORS),
        (f"/workspace/nogal-steering/optimization/{persona}.report.json", OPT_DIR),
    ]:
        subprocess.run([
            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
            f"root@{ssh_host}:{src}",
            str(dst),
        ], check=True)
        print(f"  {dst / Path(src).name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona")
    p.add_argument("--gpu", default="H100", choices=list(GPU_TYPES.keys()))
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--model", default=None, help="Override target_model.prod")
    p.add_argument("--keep-alive", action="store_true", help="No terminar pod al finalizar")
    args = p.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY no seteado en .env")
        sys.exit(1)

    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    persona_yaml = REPO_ROOT / "personas" / f"{args.persona}.yaml"
    with open(persona_yaml) as f:
        persona_data = yaml.safe_load(f)

    model_id = args.model or persona_data["target_model"]["prod"]
    pod_name = f"steering-{args.persona}-{int(time.time())}"

    pod = None
    try:
        pod = create_pod(pod_name, args.gpu)
        rsync_to_pod(pod, args.persona, REPO_ROOT)
        run_pipeline_remote(pod, args.persona, args.n_trials, model_id)
        download_results(pod, args.persona, REPO_ROOT)

        print(f"\nListo. Vector: vectors/{args.persona}.gguf")

    finally:
        if pod and not args.keep_alive:
            print(f"\nTerminando pod {pod['id']}...")
            runpod.terminate_pod(pod["id"])
            print("  pod terminado")
        elif pod:
            print(f"\nPod {pod['id']} sigue activo (--keep-alive)")


if __name__ == "__main__":
    main()
