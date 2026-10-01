"""
CLI: same Heretic job on local VRAM, a VPS (SSH), or RunPod BYOK.

  python -m src.obliterate run MODEL --backend local
  python -m src.obliterate run MODEL --backend vps --host gpu.example.com --identity ~/.ssh/id_ed25519
  python -m src.obliterate run MODEL --backend runpod --gpu 4090
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.envutil import REPO_ROOT, load_env
from src.jobs.heretic import run_heretic
from src.transport.local import LocalTransport
from src.transport.runpod import (
    DEFAULT_IMAGE,
    GPU_TYPES,
    RunpodTransport,
    create_pod,
)
from src.transport.ssh import SshTransport


def main(argv=None) -> int:
    load_env()
    p = argparse.ArgumentParser(prog="obliterate")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Heretic Optuna on local GPU, VPS, or RunPod")
    run.add_argument("model")
    run.add_argument("--trials", type=int, default=15)
    run.add_argument(
        "--backend",
        choices=["local", "vps", "ssh", "runpod"],
        required=True,
        help="local = esta maquina; vps/ssh = host por SSH; runpod = alquilar GPU",
    )
    run.add_argument("--gpu", default="4090", choices=list(GPU_TYPES.keys()))
    run.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    run.add_argument("--disk", type=int, default=50)
    run.add_argument("--volume", type=int, default=50)
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--keep-alive", action="store_true")
    run.add_argument("--host", default=None)
    run.add_argument("--user", default="ubuntu")
    run.add_argument("--port", type=int, default=22)
    run.add_argument("--identity", default=None)
    run.add_argument("--remote-dir", default="/workspace/nogal-obliterate")
    run.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts" / "obliterate"),
        help="Local dir for HF download",
    )

    args = p.parse_args(argv)
    if args.cmd != "run":
        return 1

    local_out = Path(args.out) / args.model.replace("/", "--")
    transport = None
    try:
        if args.backend == "runpod":
            pod = create_pod(
                name=f"obliterate-{int(time.time())}",
                gpu=args.gpu,
                image=args.image,
                cloud=args.cloud,
                disk=args.disk,
                volume=args.volume,
            )
            transport = RunpodTransport(pod, keep_alive=args.keep_alive)
            remote_dir = "/workspace/nogal-obliterate"
        elif args.backend == "local":
            transport = LocalTransport()
            remote_dir = args.remote_dir
            if remote_dir == "/workspace/nogal-obliterate":
                remote_dir = str(REPO_ROOT / "artifacts" / "obliterate-work")
        else:
            if not args.host:
                raise SystemExit("--host requerido con --backend vps/ssh")
            transport = SshTransport(
                host=args.host,
                user=args.user,
                port=args.port,
                identity=args.identity,
            )
            remote_dir = args.remote_dir
        run_heretic(
            transport,
            model=args.model,
            trials=args.trials,
            remote_dir=remote_dir,
            local_out=local_out,
        )
    finally:
        if transport is not None:
            transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
