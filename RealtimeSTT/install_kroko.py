"""Build and install Kroko-ONNX for the active RealtimeSTT environment."""

from __future__ import print_function

import argparse
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_REPO = "https://github.com/kroko-ai/kroko-onnx.git"
DEFAULT_BRANCH = "cross-platform-builds"
SUPPORTED_VARIANTS = ("free", "pro")
KROKO_LICENSE_QUIET_ENV = "KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT"


class KrokoInstallError(RuntimeError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stt-install-kroko",
        description=(
            "Build and install Kroko-ONNX for the active Python environment. "
            "Windows builds a wheel with Kroko's Docker workflow; Linux installs "
            "from the upstream source checkout."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build/install Kroko-ONNX from the upstream source checkout.",
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="free",
        help="Build the free community runtime or the licensed pro runtime.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Kroko-ONNX git repository URL.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Kroko-ONNX git branch to build.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory used for the Kroko-ONNX checkout and build artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing builder checkout before cloning again.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Build only; do not install the produced package into this Python.",
    )
    return parser.parse_args(argv)


def quote_cmd(cmd):
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd, cwd=None, env=None):
    print("+ " + quote_cmd(cmd))
    try:
        subprocess.check_call(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise KrokoInstallError(
            "Command failed with exit code {0}: {1}".format(
                exc.returncode,
                quote_cmd(cmd),
            )
        )


def ensure_program(name, message):
    if shutil.which(name) is None:
        raise KrokoInstallError(message)


def default_work_dir():
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "RealtimeSTT" / "kroko-builder"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "RealtimeSTT" / "kroko-builder"
    else:
        root = os.environ.get("XDG_CACHE_HOME")
        if root:
            return Path(root) / "realtimestt" / "kroko-builder"
    return Path(tempfile.gettempdir()) / "realtimestt-kroko-builder"


def remove_tree_inside(path, root):
    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise KrokoInstallError("Refusing to remove path outside builder cache: {0}".format(path))

    def clear_readonly(func, failed_path, _exc_info):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(str(path), onerror=clear_readonly)


def prepare_checkout(args):
    ensure_program("git", "Git is required to download Kroko-ONNX.")

    work_dir = (args.work_dir or default_work_dir()).expanduser().resolve()
    repo_dir = work_dir / "kroko-onnx"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.force and repo_dir.exists():
        print("Removing existing Kroko-ONNX checkout: {0}".format(repo_dir))
        remove_tree_inside(repo_dir, work_dir)

    if not repo_dir.exists():
        run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.repo,
                str(repo_dir),
            ]
        )
    else:
        print("Using existing Kroko-ONNX checkout: {0}".format(repo_dir))
        print("Pass --force to delete and clone it again.")

    return repo_dir


def read_text(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def write_text(path, text):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def normalize_lf(path):
    data = path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    if normalized != data:
        path.write_bytes(normalized)
        print("Normalized LF line endings: {0}".format(path.name))


def sanitize_batch_ascii(path):
    text = read_text(path)
    sanitized = "".join(char if ord(char) < 128 else "-" for char in text)
    if sanitized != text:
        with path.open("w", encoding="ascii", newline="") as handle:
            handle.write(sanitized)
        print("Normalized build_windows.bat to ASCII for cmd.exe.")


def patch_windows_bat(repo_dir):
    path = repo_dir / "build_windows.bat"
    if not path.exists():
        raise KrokoInstallError("Missing Kroko Windows build script: {0}".format(path))

    text = read_text(path)
    if 'findstr /C:"set(SHERPA_ONNX_VERSION"' in text:
        return
    if "Select-String" not in text or "SHERPA_ONNX_VERSION" not in text:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    start = text.find('REM CMakeLists has:  set(SHERPA_ONNX_VERSION "1.12.9")')
    if start == -1:
        start = text.find("set \"VERSION=\"")
    if start == -1:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    if_pos = text.find('if "%VERSION%"==""', start)
    if if_pos == -1:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    block_end = text.rfind("\n", 0, if_pos) + 1
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement = newline.join(
        [
            'REM CMakeLists has:  set(SHERPA_ONNX_VERSION "1.12.9")',
            "REM Keep this pure batch so cmd.exe does not parse nested PowerShell regex",
            "REM parentheses inside a FOR command substitution.",
            'set "VERSION="',
            'for /f "tokens=2 delims= " %%v in (\'findstr /C:"set(SHERPA_ONNX_VERSION" "%ROOT%\\CMakeLists.txt"\') do set "VERSION=%%~v"',
            'set "VERSION=%VERSION:"=%"',
            'set "VERSION=%VERSION:)=%"',
            "",
        ]
    )
    write_text(path, text[:start] + replacement + text[block_end:])
    print("Patched build_windows.bat version parsing for cmd.exe.")


def patch_windows_dockerfile(repo_dir):
    path = repo_dir / "Dockerfile.windows"
    if not path.exists():
        raise KrokoInstallError("Missing Kroko Windows Dockerfile: {0}".format(path))

    text = read_text(path)
    if "sed -i 's/\\r$//'" in text:
        return

    old_lf = (
        "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
        "RUN chmod +x /usr/local/bin/in_windows_container.sh"
    )
    new_lf = (
        "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
        "RUN sed -i 's/\\r$//' /usr/local/bin/in_windows_container.sh \\\n"
        " && chmod +x /usr/local/bin/in_windows_container.sh"
    )
    old_crlf = old_lf.replace("\n", "\r\n")
    new_crlf = new_lf.replace("\n", "\r\n")
    if old_lf in text:
        write_text(path, text.replace(old_lf, new_lf))
        print("Patched Dockerfile.windows to tolerate CRLF shell scripts.")
    elif old_crlf in text:
        write_text(path, text.replace(old_crlf, new_crlf))
        print("Patched Dockerfile.windows to tolerate CRLF shell scripts.")


def _insert_after_line(text, line_text, insertion):
    lines = text.splitlines(True)
    for index, line in enumerate(lines):
        if line.strip() == line_text:
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            if insertion in text:
                return text
            lines.insert(index + 1, insertion.replace("\n", newline))
            return "".join(lines)
    return text


def _wrap_license_output_line(text, marker):
    lines = text.splitlines(True)
    changed = False
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        if "std::cout" not in line and "std::cerr" not in line:
            continue
        previous = "".join(lines[max(0, index - 2):index])
        if "KrokoSuppressLicenseOutput" in previous:
            continue

        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        content = line.rstrip("\r\n")
        indent = content[:len(content) - len(content.lstrip())]
        statement = content[len(indent):]
        lines[index] = (
            indent + "if (!KrokoSuppressLicenseOutput()) {" + newline
            + indent + "    " + statement + newline
            + indent + "}" + newline
        )
        changed = True

    if not changed:
        return text
    return "".join(lines)


def patch_license_quiet_env(repo_dir):
    path = repo_dir / "sherpa-onnx" / "csrc" / "license.h"
    if not path.exists():
        print("Could not find Kroko license client source; native license logs may remain noisy.")
        return

    text = read_text(path)
    original = text

    if "#include <cstdlib>" not in text:
        text = _insert_after_line(text, "#include <chrono>", "#include <cstdlib>\n")
    if "#include <windows.h>" not in text:
        text = _insert_after_line(
            text,
            "#include <cstdlib>",
            "#ifdef _WIN32\n#include <windows.h>\n#endif\n",
        )

    helper = (
        "inline std::string KrokoLicenseQuietEnvValue() {\n"
        "#ifdef _WIN32\n"
        "    char buffer[64];\n"
        "    DWORD size = GetEnvironmentVariableA(\n"
        "        \"" + KROKO_LICENSE_QUIET_ENV + "\",\n"
        "        buffer,\n"
        "        static_cast<DWORD>(sizeof(buffer)));\n"
        "    if (size > 0) {\n"
        "        if (size < sizeof(buffer)) {\n"
        "            return std::string(buffer, size);\n"
        "        }\n"
        "        return \"1\";\n"
        "    }\n"
        "#endif\n"
        "\n"
        "    const char* value = std::getenv(\"" + KROKO_LICENSE_QUIET_ENV + "\");\n"
        "    if (value == nullptr) {\n"
        "        return \"\";\n"
        "    }\n"
        "    return std::string(value);\n"
        "}\n\n"
        "inline bool KrokoSuppressLicenseOutput() {\n"
        "    std::string text = KrokoLicenseQuietEnvValue();\n"
        "    return !text.empty() && text != \"0\" && text != \"false\" && text != \"False\" && text != \"FALSE\";\n"
        "}\n\n"
    )
    helper_start = text.find("inline bool KrokoSuppressLicenseOutput() {")
    if helper_start != -1:
        block_start = text.rfind("\ninline ", 0, helper_start)
        if block_start == -1:
            block_start = helper_start
        else:
            block_start += 1
        block_end = text.find("struct Feature {", helper_start)
        if block_end != -1:
            text = text[:block_start] + helper + text[block_end:]
    elif KROKO_LICENSE_QUIET_ENV not in text:
        text = _insert_after_line(text, "using json = nlohmann::json;", "\n" + helper)

    for marker in (
        "License not allowed:",
        "License accepted. Remaining seconds:",
        "Usage report error:",
        "Remaining seconds updated:",
        "JSON parse error:",
        "Connected to license server.",
        "Connection closed.",
        "Connection failed.",
        "Failed to create connection:",
        "Retrying connection in 3s...",
        "Cannot send usage: license not allowed.",
        "No active WebSocket connection.",
        "Failed to send usage report:",
        "Offline timeout exceeded (",
    ):
        text = _wrap_license_output_line(text, marker)

    if text != original:
        write_text(path, text)
        print("Patched Kroko license client to honor {0}.".format(KROKO_LICENSE_QUIET_ENV))


def prepare_windows_checkout(repo_dir):
    script = repo_dir / "in_windows_container.sh"
    if not script.exists():
        raise KrokoInstallError("Missing Kroko container build script: {0}".format(script))
    normalize_lf(script)
    patch_windows_bat(repo_dir)
    sanitize_batch_ascii(repo_dir / "build_windows.bat")
    patch_windows_dockerfile(repo_dir)
    patch_license_quiet_env(repo_dir)


def ensure_windows_host():
    if sys.version_info[:2] != (3, 12):
        raise KrokoInstallError(
            "Kroko's current Windows wheel build targets CPython 3.12. "
            "Run this command from a Python 3.12 x64 environment."
        )
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        raise KrokoInstallError(
            "Kroko's current Windows wheel build targets win_amd64; "
            "this machine reports {0}.".format(platform.machine())
        )
    ensure_program("docker", "Docker Desktop is required on Windows. Start Docker Desktop and try again.")
    try:
        run(["docker", "version"])
    except KrokoInstallError:
        raise KrokoInstallError(
            "Docker Desktop must be running on Windows before building Kroko. "
            "Start Docker Desktop with the WSL2 backend enabled, then retry."
        )


def find_windows_wheel(repo_dir, variant):
    tag = "cp{0}{1}".format(sys.version_info.major, sys.version_info.minor)
    wheel_dir = repo_dir / "release_artifacts" / "windows"
    patterns = [
        "kroko_onnx-*-1{0}-{1}-{1}-win_amd64.whl".format(variant, tag),
        "kroko_onnx-*-{0}-{1}-{1}-win_amd64.whl".format(variant, tag),
        "kroko_onnx-*-{0}-{0}-win_amd64.whl".format(tag),
    ]
    wheels = []
    for pattern in patterns:
        wheels.extend(wheel_dir.glob(pattern))
    wheels = sorted(set(wheels), key=lambda item: item.stat().st_mtime, reverse=True)
    if not wheels:
        raise KrokoInstallError(
            "Windows build finished, but no Kroko wheel matching {0}/{1} was found in {2}.".format(
                variant,
                tag,
                wheel_dir,
            )
        )
    return wheels[0]


def install_windows(args, repo_dir):
    ensure_windows_host()
    prepare_windows_checkout(repo_dir)
    run(["cmd.exe", "/c", str(repo_dir / "build_windows.bat"), "--variant", args.variant], cwd=repo_dir)
    wheel = find_windows_wheel(repo_dir, args.variant)
    print("Built Kroko-ONNX wheel: {0}".format(wheel))
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)])


def install_linux(args, repo_dir):
    ensure_program("cmake", "CMake is required to build Kroko-ONNX from source on Linux.")
    patch_license_quiet_env(repo_dir)
    env = os.environ.copy()
    if args.variant == "pro":
        env["KROKO_LICENSE"] = "ON"

    if args.skip_install:
        wheel_dir = repo_dir / "release_artifacts" / "linux"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
            ],
            cwd=repo_dir,
            env=env,
        )
        return

    run([sys.executable, "-m", "pip", "install", "."], cwd=repo_dir, env=env)


def main(argv=None):
    args = parse_args(argv)
    if not args.build:
        raise SystemExit("Pass --build to build and install Kroko-ONNX.")

    try:
        repo_dir = prepare_checkout(args)
        if os.name == "nt":
            install_windows(args, repo_dir)
        elif sys.platform.startswith("linux"):
            install_linux(args, repo_dir)
        else:
            raise KrokoInstallError(
                "stt-install-kroko currently supports Windows and Linux. "
                "Use Kroko's upstream macOS build script on macOS."
            )
    except KrokoInstallError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 1

    print("Kroko-ONNX is ready in this Python environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
