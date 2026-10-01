"""RunPod REST v2: create pod, SSH direct, DELETE on close. BYOK via RUNPOD_API_KEY."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from src.transport.ssh import SshTransport

API = "https://api.runpod.io"

GPU_TYPES = {
    "H100": "NVIDIA H100 80GB HBM3",
    "H100PCIE": "NVIDIA H100 PCIe",
    "A100": "NVIDIA A100 80GB PCIe",
    "L40S": "NVIDIA L40S",
    "A6000": "NVIDIA RTX A6000",
    "4090": "NVIDIA GeForce RTX 4090",
}

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def api(method: str, path: str, body=None, timeout=60):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY no seteado (BYOK: .env o export)")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "nogal-steering/obliterate",
        },
    )
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                problem = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                problem = {"detail": raw, "status": e.code}
            detail = problem.get("detail") or problem.get("title") or raw
            if e.code == 429:
                retry_after = e.headers.get("Retry-After") or str(2 ** attempt)
                time.sleep(int(retry_after))
                last_err = f"429 {detail}"
                continue
            if e.code >= 500:
                time.sleep(2 ** attempt)
                last_err = f"{e.code} {detail}"
                continue
            raise RuntimeError(f"RunPod {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"RunPod request failed after retries: {last_err}")


def ssh_direct(pod: dict) -> tuple[str, int, str]:
    ssh = pod.get("ssh") or {}
    direct = ssh.get("direct")
    if not isinstance(direct, dict):
        raise RuntimeError(
            "Pod sin ssh.direct. startSsh=true, 22/tcp, SSH key en la cuenta RunPod."
        )
    host = direct.get("host")
    port = direct.get("port")
    if not host or port is None:
        raise RuntimeError("Pod sin ssh.direct host/port")
    return host, int(port), direct.get("username") or "root"


def create_pod(
    name: str,
    gpu: str,
    image: str,
    cloud: str,
    disk: int,
    volume: int,
) -> dict:
    gpu_id = GPU_TYPES.get(gpu, gpu)
    print(f"Creando pod {name} con {gpu_id} ({cloud})...")
    body = {
        "name": name,
        "image": image,
        "cloud": cloud,
        "gpu": {"id": gpu_id, "count": 1},
        "disk": disk,
        "ports": ["22/tcp", "8888/http"],
        "startSsh": True,
        "mounts": {"persistent": {"size": volume, "path": "/workspace"}},
    }
    pod = api("POST", "/v2/pods", body)
    if not pod or "id" not in pod:
        raise RuntimeError("POST /v2/pods no devolvio id")
    pod_id = pod["id"]
    print(f"  pod_id: {pod_id}  status={pod.get('status')}")
    print("  esperando RUNNING...", end="", flush=True)
    for _ in range(60):
        time.sleep(5)
        info = api("GET", f"/v2/pods/{pod_id}")
        if not info:
            print("?", end="", flush=True)
            continue
        status = info.get("status")
        print(".", end="", flush=True)
        if status == "RUNNING":
            if (info.get("ssh") or {}).get("direct"):
                print(" OK")
                return info
            print("s", end="", flush=True)
            continue
        if status in ("ERROR", "TERMINATED", "EXITED"):
            raise RuntimeError(f"Pod {pod_id} termino en {status}")
    raise TimeoutError("Pod no arranco en 5 min")


class RunpodTransport:
    def __init__(self, pod: dict, keep_alive: bool = False):
        host, port, user = ssh_direct(pod)
        self.pod_id = pod["id"]
        self.keep_alive = keep_alive
        self._ssh = SshTransport(host=host, user=user, port=port)

    def ssh(self, remote_cmd: str) -> None:
        self._ssh.ssh(remote_cmd)

    def rsync_up(self, local_dir: str, remote_dir: str) -> None:
        self._ssh.rsync_up(local_dir, remote_dir)

    def rsync_down(self, remote_path: str, local_dir: str) -> None:
        self._ssh.rsync_down(remote_path, local_dir)

    def close(self) -> None:
        if self.keep_alive:
            print(f"Pod {self.pod_id} sigue activo (--keep-alive)")
            return
        print(f"Terminando pod {self.pod_id}...")
        api("DELETE", f"/v2/pods/{self.pod_id}")
        print("  pod terminado")
