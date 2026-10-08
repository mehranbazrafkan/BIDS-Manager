"""Lifecycle of the bundled AI agent service.

``bidsmgr/ai_agent`` is a Flask app carrying its own stack — torch,
transformers, laya, bitsandbytes — none of which BIDS Manager ships,
because a multi-gigabyte model and its runtime have no business being a
dependency of a file manager. So the agent is a *separate process* and
this module is what starts and stops it.

Two things make that awkward enough to be worth writing down.

**The interpreter that runs this app is usually not the one that can run
the agent.** The venv BIDS Manager is installed into rarely has ``torch``
(it is a GUI, not a model runtime). Candidates are therefore probed
without importing anything — :func:`importlib.util.find_spec` is enough
— and the first interpreter that can see ``flask``, ``torch``,
``transformers`` and ``laya`` wins. A user can pin one explicitly in
Settings → AI Agent.

**Port 8000 is a crowded address.** Occupancy is settled by *binding*
(:func:`port_is_free`) — a connection cannot tell a free port from a slow
one, and asking costs a timeout on the one path where seconds matter —
and only then is ``GET /models`` asked who is actually there:

* free            → start on 8000
* an agent        → reuse it, and leave it alone on exit, because it is
                    not ours to kill
* something else  → pick a free port and tell the agent about it through
                    ``BIDS_AGENT_PORT`` (it reads that at startup)

The bind address is forced to loopback as well: a service we started is
a service only this machine should reach, and ``0.0.0.0`` was chosen for
a developer running it by hand, not for a desktop app spawning it.

Everything here is stdlib. The GUI reads :class:`AgentService`'s
properties from a :class:`~PyQt6.QtCore.QTimer` rather than through
listeners, so no Qt ever has to be touched from a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: Where the agent's ``main.py`` lives, relative to this module.
AGENT_DIR = Path(__file__).resolve().parent / "ai_agent"

#: Loopback only — see the module docstring.
HOST = "127.0.0.1"

#: The agent's own default. Its ``app.run`` reads ``BIDS_AGENT_PORT`` so
#: this can be moved when something else already owns 8000.
DEFAULT_PORT = 8000

#: What an interpreter must be able to see before it can serve.
REQUIRED_MODULES = ("flask", "torch", "transformers", "laya")

#: Generous: importing torch + transformers + laya on a laptop is slow,
#: and a first run may also be pulling a model into the cache.
START_TIMEOUT = 90.0

#: Probe interpreter candidates with this; it never imports the heavy
#: modules, so it costs a process start and nothing more.
_PROBE_PY = (
    "import importlib.util as u, sys\n"
    "def _has(name):\n"
    "    try:\n"
    "        return u.find_spec(name) is not None\n"
    "    except Exception:\n"
    "        return False\n"
    "names = ('flask', 'torch', 'transformers', 'laya')\n"
    "sys.exit(0 if all(_has(n) for n in names) else 1)\n"
)

_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
)


def default_url() -> str:
    """The address the agent occupies when nothing has moved it.

    Read as a module global at call time so a test can point the whole
    service at a throwaway port without patching an instance.
    """
    return f"http://{HOST}:{DEFAULT_PORT}"


# =====================================================================
# Probing
# =====================================================================

def probe(url: str, timeout: float = 4.0) -> str:
    """``"agent"`` if the BIDS AI agent is answering at ``url``, else ``"no"``.

    Deliberately does not try to say *why* it is not answering. Telling a
    free port from an occupied one by connecting does not work: a refused
    connection, a reset, a firewall silently dropping the SYN and Windows
    loopback's habit of completing a socket to its own ephemeral port and
    then hanging until the read times out all arrive here looking the
    same. Only ``bind()`` distinguishes them, which is what
    :func:`port_is_free` is for. This function answers one question —
    "is that our agent?" — and nothing else.
    """
    request = urllib.request.Request(
        f"{url.rstrip('/')}/models",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read(65536)
    except Exception:                                 # noqa: BLE001
        return "no"
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return "no"
    return "agent" if isinstance(data, dict) and "models" in data else "no"


def _can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def port_is_free(port: int) -> bool:
    """Whether *we* could bind ``port`` right now.

    Ask by binding rather than by connecting — see :func:`probe` for why
    connecting lies. ``SO_REUSEADDR`` is deliberately off: on Windows that
    flag lets a socket bind straight over a *live* listener, which would
    turn this test into a permanent "free".

    Two addresses are tried, because Windows does not consider them
    interchangeable: a listener on ``127.0.0.1`` does not block a bind on
    ``0.0.0.0`` and vice versa, so either one failing means somebody has
    the port. (Measured, not assumed — the wildcard bind happily succeeds
    over the loopback listener.)
    """
    return _can_bind(HOST, port) and _can_bind("", port)


def free_port() -> int:
    """A port on loopback that was free a moment ago.

    Bind-then-release is the only portable way to ask; the window between
    releasing and the child binding is small and, if somebody wins it,
    the failure is reported rather than silently ignored.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _candidates(hint: str) -> list[list[str]]:
    """Interpreter commands to try, most likely first."""
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(*cmd: str) -> None:
        if not cmd or not cmd[0]:
            return
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            out.append(list(cmd))

    if hint.strip():
        add(hint.strip())
    # The app's own interpreter first: if it can run the agent, it is by
    # far the least surprising choice.
    add(sys.executable)
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            add(found)
    if os.name == "nt":
        # The launcher reaches Pythons that are not on PATH at all — the
        # common case on a machine where BIDS Manager came from an
        # installer and the agent's stack lives in a system Python.
        found = shutil.which("py")
        if found:
            add(found, "-3")
    return out


def _can_run(cmd: list[str], *, timeout: float) -> bool:
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.run([*cmd, "-c", _PROBE_PY], **kwargs).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_python(hint: str = "", *, timeout: float = 40.0) -> Optional[list[str]]:
    """First interpreter that can see everything the agent imports."""
    for candidate in _candidates(hint):
        if _can_run(candidate, timeout=timeout):
            return candidate
    return None


def _log_path() -> Path:
    return Path(tempfile.gettempdir()) / "bidsmgr-ai-agent.log"


def _tail(path: Path, chars: int = 800) -> str:
    try:
        raw = path.read_bytes()[-chars:]
    except OSError:
        return ""
    return raw.decode("utf-8", "replace").strip()


def _terminate(proc: subprocess.Popen, grace: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=grace)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
            proc.wait(timeout=grace)
        except (subprocess.TimeoutExpired, OSError):
            log.warning("agent pid %s would not die", proc.pid)


# =====================================================================
# Messages that are meant for a person
# =====================================================================

class _Failure(RuntimeError):
    """A start attempt that ended in a sentence a user can act on."""


def _no_python_message(hint: str) -> str:
    text = (
        "Could not find a Python that has everything the AI agent needs:\n"
        f"    {', '.join(REQUIRED_MODULES)}"
    )
    if hint:
        text += (
            "\n\nThe interpreter chosen in Settings -> AI Agent was:\n"
            f"    {hint}\n"
            "and it does not have all of them."
        )
    text += (
        "\n\nInstall them into whichever Python runs the agent:\n"
        "    pip install flask torch transformers laya\n"
        "or pick that interpreter under Settings -> AI Agent."
    )
    return text


def _spawn_failure(log_path: Path, url: str) -> str:
    tail = _tail(log_path)
    if "Address already in use" in tail or "WinError 10048" in tail:
        return (
            f"Something else claimed {url} before the agent could bind it.\n"
            "Close whatever is using that port and press Start again."
        )
    missing = _MISSING_MODULE_RE.search(tail)
    if missing:
        name = missing.group(1).split(".")[0]
        return (
            f"The agent's Python is missing {name!r}.\n\n"
            "Install it with:\n"
            f"    pip install {name}\n"
            "or choose a different interpreter in Settings -> AI Agent."
        )
    if tail:
        return f"The AI agent exited before it was ready:\n\n{tail}"
    return "The AI agent exited before it was ready."


def _call(fn: Optional[Callable[[], None]]) -> None:
    if fn is None:
        return
    try:
        fn()
    except Exception:                                 # pragma: no cover
        log.exception("agent on_ready callback failed")


# =====================================================================
# The service
# =====================================================================

class AgentService:
    """Starts, observes and stops one agent process.

    States: ``stopped`` / ``starting`` / ``running`` / ``failed``. Every
    mutation happens under one lock and is tagged with a generation, so
    a ``stop()`` that lands while a ``start()`` is still working can never
    leave a stray process behind — the start notices it has been superseded
    and tears down whatever it had got as far as launching.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "stopped"
        self._detail = "Not running."
        self._url = ""
        self._owned = False
        self._proc: Optional[subprocess.Popen] = None
        self._generation = 0

    # -- observation -------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    @property
    def url(self) -> str:
        """Where to talk to it, or ``""`` when it is not running."""
        with self._lock:
            return self._url

    @property
    def owned(self) -> bool:
        """True when *we* started it, i.e. when we are the ones who stop it."""
        with self._lock:
            return self._owned

    @property
    def log_path(self) -> Path:
        return _log_path()

    def is_running(self) -> bool:
        return self.state == "running"

    # -- control -----------------------------------------------------

    def start(
        self,
        *,
        python: str = "",
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        """Bring the agent up in the background. Never raises, never blocks.

        Returns immediately; poll :attr:`state` (the Settings tab does,
        four times a second) or use :meth:`wait_running`.
        """
        with self._lock:
            if self._state in ("starting", "running"):
                return
            self._generation += 1
            generation = self._generation
            self._state = "starting"
            self._detail = "Starting the AI agent..."
        threading.Thread(
            target=self._run,
            args=(generation, python, on_ready),
            name="bidsmgr-ai-agent",
            daemon=True,
        ).start()

    def stop(self) -> None:
        """Stop the agent, but only the one we started.

        An agent somebody launched by hand is left serving: they may have
        other things pointed at it, and it was not ours to begin with.
        """
        with self._lock:
            self._generation += 1          # supersede a start in flight
            proc, owned = self._proc, self._owned
            if self._state == "running" and not owned:
                return                     # foreign agent: leave it alone
            self._proc = None
            self._owned = False
            self._state = "stopped"
            self._detail = "Not running."
            self._url = ""
        if owned and proc is not None:
            _terminate(proc)

    def wait_running(self, timeout: float = 60.0) -> bool:
        """Block until it is serving, or it is clear it will not be.

        Used by the Ask AI worker so the first click of a session does not
        race the import of torch on the agent's side.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            state = self.state
            if state == "running":
                return True
            if state != "starting" or time.monotonic() >= deadline:
                return state == "running"
            time.sleep(0.25)

    # -- internals ---------------------------------------------------

    def _publish(
        self,
        generation: int,
        state: str,
        detail: str,
        url: str = "",
        owned: bool = False,
    ) -> bool:
        """Record a new state, unless a stop() has superseded us."""
        with self._lock:
            if generation != self._generation:
                return False
            self._state = state
            self._detail = detail
            self._url = url
            self._owned = owned
        return True

    def _stale(self, generation: int) -> bool:
        with self._lock:
            return generation != self._generation

    def _run(
        self,
        generation: int,
        hint: str,
        on_ready: Optional[Callable[[], None]],
    ) -> None:
        log_path = _log_path()
        try:
            # Something may already own 8000 — ask with a bind first: a
            # connect cannot tell a free port from a slow one (see probe),
            # and probing a free port would otherwise cost a full timeout
            # on every launch, on a path where every second counts.
            existing = default_url()
            if port_is_free(DEFAULT_PORT):
                port = DEFAULT_PORT
            else:
                if probe(existing) == "agent":
                    # Somebody's agent, not ours: reuse it, and leave it
                    # alone on exit for the same reason.
                    if self._publish(
                        generation,
                        "running",
                        "Already running (started outside BIDS-Manager).",
                        existing,
                        owned=False,
                    ):
                        _call(on_ready)
                    return
                port = free_port()
            if self._stale(generation):
                return

            if not (AGENT_DIR / "main.py").is_file():
                raise _Failure(
                    "The AI agent's code was not found at\n"
                    f"    {AGENT_DIR}\n"
                    "Reinstall BIDS-Manager, or copy bids-ai-agent there."
                )

            exe = resolve_python(hint)
            if exe is None:
                raise _Failure(_no_python_message(hint))

            proc = self._spawn(exe, port, log_path, generation)
            if proc is None:
                return                     # stopped underneath us
            self._await_serving(
                proc, f"http://{HOST}:{port}", port, log_path, generation,
            )
            if self._stale(generation):
                return
            _call(on_ready)
        except _Failure as exc:
            if not self._stale(generation):
                self._publish(generation, "failed", str(exc))
                log.warning("AI agent did not start: %s", exc)
        except Exception as exc:                          # pragma: no cover
            if not self._stale(generation):
                self._publish(
                    generation, "failed",
                    f"Could not start the AI agent: {exc}",
                )
                log.exception("AI agent start failed")

    def _spawn(
        self,
        exe: list[str],
        port: int,
        log_path: Path,
        generation: int,
    ) -> Optional[subprocess.Popen]:
        env = dict(os.environ)
        env["BIDS_AGENT_PORT"] = str(port)
        env["BIDS_AGENT_HOST"] = HOST
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            handle = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:                                   # pragma: no cover
            handle = None

        kwargs: dict = {}
        if os.name == "nt":
            # No console flash, and a process group so a stray reloader
            # (if debug were ever turned on) could not outlive us.
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(
                [*exe, "main.py"],
                cwd=str(AGENT_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle if handle is not None else subprocess.DEVNULL,
                stderr=(
                    subprocess.STDOUT
                    if handle is not None
                    else subprocess.DEVNULL
                ),
                **kwargs,
            )
        except OSError as exc:
            raise _Failure(
                f"Could not launch {exe[0]}:\n    {exc}"
            ) from exc
        finally:
            if handle is not None:
                handle.close()      # the child holds its own copy

        with self._lock:
            stale = generation != self._generation
            if not stale:
                self._proc = proc
                self._owned = True
        if stale:
            _terminate(proc)
            return None
        return proc

    def _await_serving(
        self,
        proc: subprocess.Popen,
        url: str,
        port: int,
        log_path: Path,
        generation: int,
    ) -> None:
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if self._stale(generation):
                return
            if proc.poll() is not None:
                raise _Failure(_spawn_failure(log_path, url))
            # While torch is being imported there is no socket to talk
            # to, and asking an unbound port an HTTP question costs a
            # whole timeout. A bind answers in one syscall instead.
            if not port_is_free(port) and probe(url, timeout=2.0) == "agent":
                self._publish(
                    generation, "running", "Running (started by BIDS-Manager).",
                    url, owned=True,
                )
                return
            time.sleep(0.3)
        _terminate(proc)
        raise _Failure(
            f"The AI agent did not answer within {int(START_TIMEOUT)} seconds.\n"
            f"Details are in {log_path}"
        )


_SERVICE = AgentService()


def service() -> AgentService:
    """The process-wide agent, created on first use."""
    return _SERVICE


__all__ = [
    "AGENT_DIR",
    "AgentService",
    "DEFAULT_PORT",
    "HOST",
    "REQUIRED_MODULES",
    "START_TIMEOUT",
    "default_url",
    "free_port",
    "port_is_free",
    "probe",
    "resolve_python",
    "service",
]
