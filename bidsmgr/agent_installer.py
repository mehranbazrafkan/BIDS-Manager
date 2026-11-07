"""Installing the AI Agent module: clone its repository, set up its deps.

The agent is a separate program with its own heavy dependencies (torch,
transformers, laya), so it is not shipped inside BIDS-Manager: it is
cloned from its own repository the first time a user asks for it. This
module does that in three steps, each of which can be pointed at while
it runs (the Settings tab shows every line the installer produces):

1. **Clone** ``bids-ai-agent`` into the folder the service already looks
   in (:data:`agent_service.AGENT_DIR`). ``git`` is used when the machine
   has it; otherwise the repository is downloaded as a zip from GitHub
   and unpacked, so a user without git still gets the module.
2. **Install into BIDS-Manager's own environment.** The agent is meant
   to share the interpreter that runs the app — that is the setup a user
   reproduces by hand (clone the repository, then
   ``pip install -r requirements.txt`` in the app's venv), and the
   service already prefers ``sys.executable`` when it can import the
   agent's stack. Only when that environment cannot take an install (a
   frozen build, or a read-only system Python) is a private ``.venv``
   created inside the clone.
3. **Install the requirements** with that interpreter. On a machine with
   an NVIDIA GPU the repository's own requirements are used as-is (the
   CUDA torch build); on a CPU-only machine the CUDA pin is swapped for
   the plain package, which is what the agent's README already tells a
   human to do in that situation.

Everything here is stdlib, and nothing here touches the GUI: the
Settings tab wraps :func:`full_install` in a worker thread and shows the
progress this module reports through its ``emit`` callback.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from . import agent_service

log = logging.getLogger(__name__)

#: The agent lives in its own repository; this is where it is cloned from.
AGENT_REPO_URL = "https://github.com/mehranbazrafkan/bids-ai-agent.git"

#: The same repository as a zip, for machines without git.
AGENT_ZIP_URL = "https://codeload.github.com/mehranbazrafkan/bids-ai-agent/zip/refs/heads/main"

#: The Python the agent's stack needs. The repository's requirements pin
#: packages (torch, transformers) that have moved past older interpreters.
MIN_PYTHON = (3, 10)

#: A CUDA torch wheel is multiple gigabytes; pip must be allowed to take
#: its time, but a hung pip must not hang the installer forever.
PIP_TIMEOUT = 3600.0

#: Progress lines the Settings tab shows; longer pip output is trimmed.
_EMIT_CHARS = 200


class InstallError(RuntimeError):
    """A user-facing installer failure: the message is shown as-is."""


def install_dir() -> Path:
    """Where the module is cloned: the folder the service looks in."""
    return agent_service.AGENT_DIR


def installed() -> bool:
    """Whether the agent's entry point is present, i.e. whether the
    module can be started without installing anything."""
    return (install_dir() / "main.py").is_file()


def venv_dir() -> Path:
    """The private environment created inside the clone. The cloned
    repository's own ``.gitignore`` covers ``.venv`` already."""
    return install_dir() / ".venv"


def venv_python(directory: Optional[Path] = None) -> Path:
    """The interpreter inside a venv, per platform."""
    base = directory if directory is not None else venv_dir()
    if os.name == "nt":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def _run(
    cmd: list[str],
    *,
    timeout: float,
    emit: Optional[Callable[[str], None]] = None,
    cwd: Optional[Path] = None,
) -> None:
    """Run a subprocess, forwarding its output line by line.

    Everything the installer runs is silent machinery from the user's
    point of view unless its output reaches the Settings tab, so every
    line is handed to *emit* as it arrives rather than buffered until
    the end.
    """
    kwargs: dict = {"timeout": timeout}
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except OSError as exc:
        raise InstallError(f"Could not run {cmd[0]}:\n    {exc}") from exc
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            text = line.rstrip()
            if text:
                if emit is not None:
                    emit(_shorten(text))
                else:
                    log.debug("%s", _shorten(text))
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise InstallError(
            f"{cmd[0]} did not finish within {int(timeout / 60)} minutes."
        ) from exc
    if code != 0:
        raise InstallError(
            f"{cmd[0]} failed (exit code {code}). See the lines above."
        )


def _shorten(text: str) -> str:
    if len(text) <= _EMIT_CHARS:
        return text
    return text[: _EMIT_CHARS - 1] + "…"


def git_available() -> bool:
    """Whether the machine can clone via git at all."""
    found = shutil.which("git")
    if not found:
        return False
    try:
        subprocess.run(
            [found, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


# ---------------------------------------------------------------------------
# Step 1: the code
# ---------------------------------------------------------------------------

def clone(emit: Optional[Callable[[str], None]] = None) -> Path:
    """Clone the agent's repository into the folder the service looks in.

    Returns the install directory. A directory left behind by a failed
    earlier attempt (no ``main.py`` inside) is replaced; a working
    install is never touched — :func:`full_install` does not even call
    this when the module is already there.
    """
    dest = install_dir()
    if dest.exists() and not (dest / "main.py").is_file():
        # A half-clone from a failed attempt: git refuses to clone into
        # a non-empty directory, so clear the debris out of the way.
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if git_available():
        if emit is not None:
            emit(f"Cloning {AGENT_REPO_URL} ...")
        _run(
            ["git", "clone", "--depth", "1", AGENT_REPO_URL, str(dest)],
            timeout=300.0,
            emit=emit,
        )
        return dest

    # No git on the machine (the common case for a plain Windows user):
    # the repository downloads as a zip just as well.
    if emit is not None:
        emit("git not found — downloading the repository as a zip ...")
    with tempfile.TemporaryDirectory(prefix="bidsmgr-agent-") as tmp:
        archive_path = Path(tmp) / "bids-ai-agent.zip"
        try:
            urllib.request.urlretrieve(AGENT_ZIP_URL, archive_path)
        except OSError as exc:
            raise InstallError(
                "Could not download the AI Agent repository:\n"
                f"    {exc}\n"
                "Check the internet connection and try again."
            ) from exc
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp)
        # GitHub zips everything under one top-level folder; its
        # *contents* are the repository.
        extracted = [
            p for p in Path(tmp).iterdir()
            if p.is_dir() and p.name.startswith("bids-ai-agent")
        ]
        if not extracted:
            raise InstallError(
                "The downloaded archive did not look like the AI Agent "
                "repository. Try again, or clone it with git."
            )
        shutil.move(str(extracted[0]), str(dest))
    if emit is not None:
        emit(f"Module downloaded to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Step 2: the private environment
# ---------------------------------------------------------------------------

def _python_version(cmd: list[str]) -> Optional[tuple[int, int]]:
    """The interpreter's (major, minor), or None when it will not say."""
    kwargs: dict = {"timeout": 20}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [*cmd, "-c", "import sys; print('%d %d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        major, minor = proc.stdout.strip().split()
        return int(major), int(minor)
    except ValueError:
        return None


def _base_candidates() -> list[list[str]]:
    """Interpreters to build the venv on, most likely first. The app's
    own interpreter first — but only when it really is a Python (a
    frozen exe answers the version probe with silence and is skipped)."""
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(*cmd: str) -> None:
        if not cmd or not cmd[0]:
            return
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            out.append(list(cmd))

    add(sys.executable)
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            add(found)
    if os.name == "nt":
        found = shutil.which("py")
        if found:
            add(found, "-3")
    return out


def pick_base_python(emit: Optional[Callable[[str], None]] = None) -> list[str]:
    """An interpreter new enough to build the agent's environment on."""
    for candidate in _base_candidates():
        version = _python_version(candidate)
        if version is None or version < MIN_PYTHON:
            continue
        return candidate
    raise InstallError(
        "No Python 3.10 or newer was found on this machine.\n"
        "The AI Agent module needs one to set up its dependencies — "
        "install Python from python.org and try again."
    )


def _writable(directory: Path) -> bool:
    """Whether a folder really accepts a new file (os.access lies about ACLs)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".bidsmgr-write-{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def app_environment_available() -> bool:
    """Whether BIDS-Manager's own environment can take the agent's deps.

    This is the setup the agent is documented for: the interpreter that
    runs the app also runs the agent, exactly what a user gets by cloning
    the repository and running ``pip install -r requirements.txt`` inside
    the app's venv. A frozen build has no site-packages to write to, and
    a read-only system Python is not ours to extend, so both fall back to
    the private environment in :func:`create_venv`.
    """
    if getattr(sys, "frozen", False):
        return False
    exe = Path(sys.executable)
    if not exe.is_file():
        return False
    version = _python_version([str(exe)])
    if version is None or version < MIN_PYTHON:
        return False
    if importlib.util.find_spec("pip") is None:
        return False
    # A venv made for this app is the user's own, disposable space; a
    # system interpreter is only used when its site-packages accept a
    # write. (os.access reports a Windows ACL-denied folder as writable,
    # so the check is a real file.)
    if sys.prefix != sys.base_prefix:
        return True
    return _writable(Path(sysconfig.get_paths()["purelib"]))


def has_nvidia_gpu() -> bool:
    """Whether the machine has an NVIDIA driver, which decides between
    the CUDA torch build and the plain one."""
    found = shutil.which("nvidia-smi")
    if not found:
        return False
    try:
        proc = subprocess.run(
            [found],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def create_venv(
    base: list[str],
    emit: Optional[Callable[[str], None]] = None,
) -> Path:
    """Create (or reuse) the agent's private environment.

    An environment left behind by a failed earlier attempt is reused:
    ``pip install`` is idempotent, so finishing a half-built venv costs
    seconds where rebuilding costs minutes.
    """
    target = venv_dir()
    if venv_python(target).is_file():
        if emit is not None:
            emit("Reusing the private Python environment from the earlier attempt.")
        return target
    if emit is not None:
        emit("Creating a private Python environment for the agent ...")
    _run([*base, "-m", "venv", str(target)], timeout=300.0, emit=emit)
    if not venv_python(target).is_file():
        raise InstallError(
            "The private Python environment could not be created.\n"
            "Try again; if it keeps failing, install Python 3.10+ from "
            "python.org and try again."
        )
    return target


# ---------------------------------------------------------------------------
# Step 3: the dependencies
# ---------------------------------------------------------------------------

def _requirements_for_machine(
    requirements: Path,
    emit: Optional[Callable[[str], None]],
) -> Path:
    """The requirements file to install from.

    With an NVIDIA driver the repository's own file is used as-is (its
    torch pin is the CUDA build). On a CPU-only machine the CUDA pin is
    swapped for the plain package and the CUDA wheel index is dropped —
    the agent's README already asks a human to do exactly this, so the
    installer does it rather than pulling gigabytes of GPU code a CPU
    machine can never use.
    """
    if has_nvidia_gpu():
        return requirements
    try:
        text = requirements.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallError(f"Could not read {requirements}:\n    {exc}") from exc
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if stripped.startswith("--find-links"):
            continue
        if stripped.startswith("torch") and "+cu" in stripped:
            # Keep the version pin and drop only the CUDA local-version
            # suffix: torch==2.14.1+cu126 -> torch==2.14.1. Losing the
            # pin as well would float torch past what transformers (and
            # the rest of the file) were tested against.
            lines.append(stripped.split("+", 1)[0])
            continue
        lines.append(line)
    tmp = Path(tempfile.gettempdir()) / "bidsmgr-agent-requirements-cpu.txt"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if emit is not None:
        emit("No NVIDIA GPU detected — installing the CPU build of torch.")
    return tmp


def _requirements_file() -> Path:
    """The cloned module's requirements, or a message saying why not."""
    requirements = install_dir() / "requirements.txt"
    if not requirements.is_file():
        raise InstallError(
            f"The cloned module has no requirements.txt at\n    {requirements}"
        )
    return requirements


def _install_into(
    python: Path,
    emit: Optional[Callable[[str], None]] = None,
) -> None:
    """Install the agent's requirements with *python*."""
    chosen = _requirements_for_machine(_requirements_file(), emit)
    if emit is not None:
        emit("Installing dependencies (this can take several minutes) ...")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
         timeout=600.0, emit=emit)
    _run([str(python), "-m", "pip", "install", "-r", str(chosen)],
         timeout=PIP_TIMEOUT, emit=emit)


def install_dependencies(
    venv: Path,
    emit: Optional[Callable[[str], None]] = None,
) -> None:
    """Install the agent's requirements into a private environment."""
    _install_into(venv_python(venv), emit)


def full_install(emit: Optional[Callable[[str], None]] = None) -> Path:
    """Install the whole module and return the interpreter to run it with.

    The dependencies go into BIDS-Manager's own environment whenever it
    can take them: that is the setup the agent is documented for, and
    the one a user reproduces by hand in the app's venv. Only a build
    with no writable environment of its own (a frozen app, a read-only
    system Python) gets the private ``.venv`` inside the clone.

    Raises :class:`InstallError` with a message meant to be shown to the
    user as-is.
    """
    if installed():
        if emit is not None:
            emit("The AI Agent module is already installed.")
    else:
        clone(emit)

    if app_environment_available():
        python = Path(sys.executable)
        if emit is not None:
            emit(
                "Installing the agent into BIDS-Manager's own Python "
                "environment ..."
            )
        _install_into(python, emit)
    else:
        venv = create_venv(pick_base_python(emit), emit)
        install_dependencies(venv, emit)
        python = venv_python(venv)

    if emit is not None:
        emit(f"AI Agent module is ready ({python})")
    return python


__all__ = [
    "AGENT_REPO_URL",
    "AGENT_ZIP_URL",
    "InstallError",
    "app_environment_available",
    "clone",
    "create_venv",
    "full_install",
    "git_available",
    "has_nvidia_gpu",
    "install_dependencies",
    "install_dir",
    "installed",
    "pick_base_python",
    "venv_dir",
    "venv_python",
]
