import unittest

from src.jobs.heretic import REMOTE_SCRIPT
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


if __name__ == "__main__":
    unittest.main()
