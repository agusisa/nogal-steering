"""Heretic job: same remote script on RunPod or SSH."""
from __future__ import annotations

import shlex
from pathlib import Path

from src.transport.base import Transport

REMOTE_SCRIPT = r"""#!/bin/bash
set -euo pipefail
MODEL="$1"
TRIALS="$2"
WORKDIR="$3"
mkdir -p "$WORKDIR/out"
cd "$WORKDIR"
python3 -m pip install -U pip heretic-llm
# primer Pareto (enter). no interactivo.
printf '\n' | heretic run "$MODEL" --trials "$TRIALS" --optimize || \
  printf '\n' | heretic "$MODEL" --trials "$TRIALS"
# copiar el HF mas reciente a out/hf si heretic dejo un dir
if [ ! -d out/hf ]; then
  found=$(find . -maxdepth 3 -name config.json -not -path './out/*' | head -1 || true)
  if [ -n "$found" ]; then
    mkdir -p out/hf
    cp -a "$(dirname "$found")/." out/hf/
  fi
fi
echo HERETIC_DONE
"""


def run_heretic(
    transport: Transport,
    model: str,
    trials: int,
    remote_dir: str,
    local_out: Path,
) -> Path:
    local_out.mkdir(parents=True, exist_ok=True)
    scripts = local_out / "_remote"
    scripts.mkdir(exist_ok=True)
    script_path = scripts / "run_heretic.sh"
    script_path.write_text(REMOTE_SCRIPT)
    transport.rsync_up(str(scripts), f"{remote_dir}/_remote")
    cmd = (
        f"bash {shlex.quote(remote_dir)}/_remote/run_heretic.sh "
        f"{shlex.quote(model)} {int(trials)} {shlex.quote(remote_dir)}"
    )
    print(f"Heretic remoto: {model} trials={trials}")
    transport.ssh(cmd)
    dest = local_out / "hf"
    transport.rsync_down(f"{remote_dir}/out/", str(local_out))
    print(f"Bajado a {local_out}")
    return dest
