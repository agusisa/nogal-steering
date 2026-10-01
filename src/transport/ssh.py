"""SSH to a machine you already have (no RunPod API). Never terminates the host."""
from __future__ import annotations

import subprocess
from pathlib import Path


def ssh_opts(port: int, identity: str | None) -> list[str]:
    opts = [
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=15",
    ]
    if identity:
        opts.extend(["-i", str(Path(identity).expanduser())])
    return opts


class SshTransport:
    def __init__(
        self,
        host: str,
        user: str = "ubuntu",
        port: int = 22,
        identity: str | None = None,
    ):
        if not host:
            raise ValueError("host vacio")
        self.host = host
        self.user = user
        self.port = port
        self.identity = identity

    @property
    def dest(self) -> str:
        return f"{self.user}@{self.host}"

    def _ssh_e(self) -> str:
        parts = ["ssh"] + ssh_opts(self.port, self.identity)
        return " ".join(parts)

    def ssh(self, remote_cmd: str) -> None:
        cmd = ["ssh", *ssh_opts(self.port, self.identity), self.dest, remote_cmd]
        subprocess.run(cmd, check=True)

    def rsync_up(self, local_dir: str, remote_dir: str) -> None:
        self.ssh(f"mkdir -p {remote_dir}")
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                self._ssh_e(),
                "--exclude=.venv",
                "--exclude=.git",
                "--exclude=.env",
                f"{local_dir.rstrip('/')}/",
                f"{self.dest}:{remote_dir.rstrip('/')}/",
            ],
            check=True,
        )

    def rsync_down(self, remote_path: str, local_dir: str) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                self._ssh_e(),
                f"{self.dest}:{remote_path}",
                local_dir,
            ],
            check=True,
        )

    def close(self) -> None:
        return
