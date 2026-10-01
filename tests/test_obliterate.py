import tempfile
import unittest
from pathlib import Path

from src.jobs.heretic import REMOTE_SCRIPT
from src.obliterate import main as obliterate_main
from src.transport.local import LocalTransport
from src.transport.ssh import ssh_opts


class TestObliterate(unittest.TestCase):
    def test_ssh_opts_with_identity(self):
        opts = ssh_opts(2222, "/tmp/id_test")
        self.assertIn("-p", opts)
        self.assertIn("2222", opts)
        self.assertIn("-i", opts)

    def test_remote_script_has_heretic_and_noninteractive(self):
        self.assertIn("heretic-llm", REMOTE_SCRIPT)
        self.assertIn("printf", REMOTE_SCRIPT)
        self.assertIn("HERETIC_DONE", REMOTE_SCRIPT)

    def test_local_copy_roundtrip(self):
        t = LocalTransport()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            src.mkdir()
            (src / "a.txt").write_text("hi")
            t.rsync_up(str(src), str(dst))
            self.assertEqual((dst / "a.txt").read_text(), "hi")
            t.ssh("true")

    def test_cli_requires_host_for_vps(self):
        with self.assertRaises(SystemExit):
            obliterate_main(["run", "Qwen/x", "--backend", "vps"])


if __name__ == "__main__":
    unittest.main()
