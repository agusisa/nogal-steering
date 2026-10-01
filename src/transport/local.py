"""This machine: CUDA, MPS, or whatever Python sees. No SSH."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class LocalTransport:
    def ssh(self, remote_cmd: str) -> None:
        subprocess.run(["bash", "-lc", remote_cmd], check=True)

    def rsync_up(self, local_dir: str, remote_dir: str) -> None:
        src = Path(local_dir)
        dst = Path(remote_dir)
        dst.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    def rsync_down(self, remote_path: str, local_dir: str) -> None:
        src = Path(remote_path)
        dst = Path(local_dir)
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(src)
        src_res, dst_res = src.resolve(), dst.resolve()
        if src_res == dst_res or dst_res in src_res.parents:
            return
        if src.is_dir():
            for item in src.iterdir():
                target = dst / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        else:
            shutil.copy2(src, dst / src.name)

    def close(self) -> None:
        return
