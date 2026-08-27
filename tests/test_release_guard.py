from __future__ import annotations

import importlib.util
import io
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "tools" / "release_guard.py"
SPEC = importlib.util.spec_from_file_location("realtime_stt_release_guard", GUARD_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"cannot load release guard from {GUARD_PATH}")
release_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_guard)


class ReleaseGuardTests(unittest.TestCase):
    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(
                f"git {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
            )

    def _init_repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        self._git(repo, "init", "--quiet")
        (repo / "demo_pkg").mkdir()
        (repo / "demo_pkg" / "__init__.py").write_bytes(b"VALUE = 'original'\n")
        self._git(repo, "add", "demo_pkg/__init__.py")
        self._git(
            repo,
            "-c",
            "user.name=release-guard-test",
            "-c",
            "user.email=release-guard-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        )
        return repo

    @staticmethod
    def _write_wheel(path: Path) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("demo_pkg/__init__.py", "VALUE = 'original'\r\n")
            archive.writestr(
                "demo-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
            )

    def test_dirty_linked_worktree_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            linked = root / "linked"
            self._git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
            try:
                (linked / "uncommitted.py").write_text("DIRTY = True\n", encoding="utf-8")
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = release_guard.main(
                        ["check-worktrees", "--repo", str(repo)]
                    )
                self.assertEqual(result, 2)
                self.assertIn("publication blocked", stderr.getvalue())
                self.assertIn("uncommitted.py", stderr.getvalue())
            finally:
                self._git(repo, "worktree", "remove", "--force", str(linked))

    def test_attest_verify_exact_runtime_then_changed_runtime_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            runtime_package = root / "runtime" / "demo_pkg"
            runtime_package.parent.mkdir()
            runtime_package.mkdir()
            (runtime_package / "__init__.py").write_bytes(b"VALUE = 'original'\r\n")
            wheel = root / "demo-1.0-py3-none-any.whl"
            self._write_wheel(wheel)
            manifest = root / "deployment.json"
            attest_args = type(
                "Args",
                (),
                {
                    "repo": repo,
                    "component": "demo",
                    "distribution": "demo",
                    "package_dir": "demo_pkg",
                    "wheel": wheel,
                    "artifact": [],
                    "runtime_package_dir": runtime_package,
                    "dependency": [],
                    "runtime_label": "test-runtime",
                    "output": manifest,
                },
            )()
            with mock.patch.object(
                release_guard.importlib.metadata,
                "version",
                return_value="1.0",
            ):
                attested = release_guard.command_attest(attest_args)
            self.assertEqual(attested["status"], "ok")
            self.assertTrue(manifest.is_file())

            verify_args = type(
                "Args",
                (),
                {
                    "repo": repo,
                    "package_dir": "demo_pkg",
                    "wheel": wheel,
                    "artifact": [],
                    "manifest": manifest,
                },
            )()
            verified = release_guard.command_verify(verify_args)
            self.assertEqual(verified["parity"], "exact")

            (runtime_package / "__init__.py").write_bytes(b"VALUE = 'changed'\n")
            with self.assertRaisesRegex(
                release_guard.GuardError, "deployed runtime and wheel differs"
            ):
                release_guard.command_attest(attest_args)


if __name__ == "__main__":
    unittest.main()
