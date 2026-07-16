"""Create isolated venvs and smoke-test RealtimeSTT install extras.

This is a release-prep helper rather than a normal unit test. It intentionally
installs the current working tree into separate virtual environments so package
metadata and optional dependencies are tested through pip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_ROOT = ROOT / ".venvs" / "install-matrix"


@dataclass(frozen=True)
class MatrixCase:
    name: str
    extras: str
    checks: tuple[str, ...]

    @property
    def install_target(self) -> str:
        if not self.extras:
            return str(ROOT)
        return f"{ROOT}[{self.extras}]"


BASE_CHECKS = (
    "dist:realtimestt",
    "module:RealtimeSTT",
    "module:RealtimeSTT.core.realtime_boundary_detector",
    "module:RealtimeSTT.core.realtime_text_stabilizer",
    "module:RealtimeSTT.transcription_engines",
)


MATRIX = (
    MatrixCase(
        name="core",
        extras="",
        checks=BASE_CHECKS
        + (
            "module:RealtimeSTT.audio_recorder",
            "dist:webrtcvad-wheels",
            "dist:soundfile",
        ),
    ),
    MatrixCase(
        name="default-faster-whisper",
        extras="default",
        checks=BASE_CHECKS
        + (
            "dist:faster-whisper",
            "module:faster_whisper",
        ),
    ),
    MatrixCase(
        name="whispercpp-openwakeword",
        extras="whisper-cpp,openwakeword",
        checks=BASE_CHECKS
        + (
            "dist:pywhispercpp",
            "module:pywhispercpp",
            "dist:openwakeword",
            "module:openwakeword",
        ),
    ),
    MatrixCase(
        name="openai-whisper-porcupine",
        extras="openai-whisper,porcupine",
        checks=BASE_CHECKS
        + (
            "dist:openai-whisper",
            "module:whisper",
            "dist:pvporcupine",
            "module:pvporcupine",
        ),
    ),
    MatrixCase(
        name="sherpa-wakewords",
        extras="sherpa-onnx,wakewords",
        checks=BASE_CHECKS
        + (
            "dist:sherpa-onnx",
            "module:sherpa_onnx",
            "dist:pvporcupine",
            "dist:openwakeword",
        ),
    ),
    MatrixCase(
        name="transformers-qwen-parakeet",
        extras="transformers,qwen,parakeet",
        checks=BASE_CHECKS
        + (
            "dist:transformers",
            "module:transformers",
            "dist:qwen-asr",
            "dist:nemo_toolkit",
            "module:nemo.collections.asr",
        ),
    ),
    MatrixCase(
        name="all",
        extras="all",
        checks=BASE_CHECKS
        + (
            "dist:faster-whisper",
            "dist:pywhispercpp",
            "dist:openai-whisper",
            "dist:sherpa-onnx",
            "dist:transformers",
            "dist:qwen-asr",
            "dist:nemo_toolkit",
            "dist:pvporcupine",
            "dist:openwakeword",
        ),
    ),
)


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def python_executable(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def check_script(checks: tuple[str, ...]) -> str:
    return """
import importlib
import importlib.metadata
import json

checks = %s
results = []
for check in checks:
    kind, value = check.split(":", 1)
    try:
        if kind == "module":
            importlib.import_module(value)
        elif kind == "dist":
            importlib.metadata.version(value)
        else:
            raise RuntimeError("unknown check kind: " + kind)
        results.append({"check": check, "ok": True})
    except Exception as exc:
        results.append({
            "check": check,
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        })

print(json.dumps(results, indent=2))
if not all(item["ok"] for item in results):
    raise SystemExit(1)
""" % json.dumps(list(checks))


def run_case(case: MatrixCase, venv_root: Path, keep: bool) -> dict[str, object]:
    venv = venv_root / case.name
    if venv.exists() and not keep:
        shutil.rmtree(venv)

    result: dict[str, object] = {
        "name": case.name,
        "extras": case.extras or "(none)",
        "venv": str(venv),
        "steps": [],
    }

    if not venv.exists():
        create = run([sys.executable, "-m", "venv", str(venv)])
        result["steps"].append({"step": "venv", "returncode": create.returncode})
        if create.returncode != 0:
            result["output"] = create.stdout
            result["ok"] = False
            return result

    py = python_executable(venv)
    commands = (
        ("upgrade-pip", [str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"]),
        ("install", [str(py), "-m", "pip", "install", case.install_target]),
        ("checks", [str(py), "-c", check_script(case.checks)]),
    )

    for step, cmd in commands:
        completed = run(cmd, cwd=Path("C:/tmp") if sys.platform == "win32" else Path("/tmp"))
        result["steps"].append({"step": step, "returncode": completed.returncode})
        if completed.returncode != 0:
            result["failed_step"] = step
            result["output"] = completed.stdout[-12000:]
            result["ok"] = False
            return result
        if step == "checks":
            result["checks_output"] = completed.stdout

    result["ok"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--case", action="append", choices=[case.name for case in MATRIX])
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    args.venv_root.mkdir(parents=True, exist_ok=True)
    selected = [case for case in MATRIX if not args.case or case.name in args.case]
    results = [run_case(case, args.venv_root, args.keep) for case in selected]

    print(json.dumps(results, indent=2))
    return 0 if all(result.get("ok") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
