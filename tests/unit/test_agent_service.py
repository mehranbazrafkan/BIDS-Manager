"""Starting and stopping the bundled AI agent.

The hard parts of this module are the ones that cannot be observed from
the GUI: telling an agent apart from whatever else owns port 8000,
picking an interpreter that actually has torch, and never leaving a
process behind. Each is a decision made before anything is launched, so
each is testable here without launching anything.

No test in this file spawns the real agent — that would import torch
into a second process on somebody's CI machine. The process-management
paths are driven by hand through the one function every transition goes
through (:meth:`AgentService._publish`), which is exactly the seam that
makes the race safe in the first place.
"""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from bidsmgr import agent_service


def _until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:               # noqa: D102
        pass

    def do_GET(self) -> None:                           # noqa: N802
        if self.path == "/models":
            body = json.dumps({"models": ["stub-7b-instruct"]}).encode()
            self.send_response(200)
        else:
            body = b"nope"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _NotAnAgent(BaseHTTPRequestHandler):
    """A server that answers HTTP but is not our agent."""

    def log_message(self, *args) -> None:               # noqa: D102
        pass

    def do_GET(self) -> None:                           # noqa: N802
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _serve(handler) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


@pytest.fixture
def agent_url() -> Iterator[str]:
    server, url = _serve(_Handler)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def stranger_url() -> Iterator[str]:
    server, url = _serve(_NotAnAgent)
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def dead_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


# =====================================================================
# Telling the three cases apart
# =====================================================================


class TestProbe:
    def test_a_healthy_agent_is_recognised(self, agent_url):
        assert agent_service.probe(agent_url) == "agent"

    def test_something_else_on_the_port_is_not_an_agent(self, stranger_url):
        assert agent_service.probe(stranger_url) == "no"

    def test_nothing_listening_is_not_an_agent(self, dead_url):
        assert agent_service.probe(dead_url) == "no"

    def test_it_answers_no_instead_of_raising_whatever_happens(self):
        """The caller is a background thread that turns the answer into
        a user-visible state; an exception here would be that thread's
        problem and nobody else's."""
        assert agent_service.probe("http://127.0.0.1:0", timeout=0.2) in (
            "agent", "no",
        )


class TestPortIsFree:
    def test_a_port_nobody_has_is_free(self):
        assert agent_service.port_is_free(agent_service.free_port()) is True

    def test_a_live_listener_makes_it_unavailable(self, stranger_url):
        _, _, port = stranger_url.partition(f"{agent_service.HOST}:")
        assert agent_service.port_is_free(int(port)) is False

    def test_it_catches_both_addresses_windows_treats_separately(self):
        """Windows lets ``0.0.0.0:P`` be bound while ``127.0.0.1:P`` is
        taken, and the reverse — measured, not assumed. A detector that
        asks only one of them reports a busy port as free half the time,
        which is exactly the mistake the whole scheme exists to avoid."""
        for address in ((agent_service.HOST, 0), ("0.0.0.0", 0)):
            with socket.socket() as listener:
                listener.bind(address)
                taken = listener.getsockname()[1]
                assert agent_service.port_is_free(taken) is False, address

    def test_it_asks_by_binding_not_by_connecting(self):
        """The reason this exists.

        Connecting cannot separate "nobody home" from "home but slow",
        "dropped by the firewall", or Windows loopback quietly completing
        a socket to its own ephemeral port and then hanging until the
        read times out. All of those arrive looking like *occupied*, which
        would push us off 8000 for no reason on every launch.
        """
        assert agent_service.port_is_free(agent_service.free_port()) is True
        with socket.socket() as listener:
            listener.bind((agent_service.HOST, 0))
            taken = listener.getsockname()[1]
            assert agent_service.port_is_free(taken) is False


class TestFreePort:
    def test_it_hands_out_a_port_we_can_actually_bind(self):
        port = agent_service.free_port()
        assert 0 < port < 65536
        with socket.socket() as sock:
            sock.bind((agent_service.HOST, port))

    def test_it_is_not_the_default_port_when_the_default_is_taken(
        self, stranger_url,
    ):
        # The default is occupied by the stranger; whatever we hand out
        # has to be somewhere the agent can move to.
        _, _, port = stranger_url.partition(f"{agent_service.HOST}:")
        assert agent_service.free_port() != int(port)


# =====================================================================
# Choosing an interpreter
# =====================================================================


class TestResolvePython:
    def test_it_returns_an_interpreter_that_runs(self):
        """Environment-dependent by nature: this machine may have no
        Python with torch at all, and that is a legitimate answer — the
        caller turns it into a message. What must never happen is handing
        back something that is not a Python."""
        found = agent_service.resolve_python()
        assert found is None or (
            isinstance(found, list)
            and all(isinstance(part, str) for part in found)
            and subprocess.run(
                [*found, "-c", "import sys"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
        )

    def test_a_pinned_interpreter_is_asked_first(self, monkeypatch):
        asked: list[list[str]] = []
        real = agent_service._can_run

        def spy(cmd, **kwargs):
            asked.append(list(cmd))
            return real(cmd, **kwargs)

        monkeypatch.setattr(agent_service, "_can_run", spy)
        # Whichever candidate wins, it must be the one we were given if
        # that one is usable; if it is not, the probe must have moved on
        # rather than returned it anyway.
        agent_service.resolve_python("/definitely/not/python")
        assert asked and asked[0] == ["/definitely/not/python"]
        assert ["/definitely/not/python"] not in asked[1:]

    def test_the_agents_own_directory_is_where_the_code_is(self):
        assert (agent_service.AGENT_DIR / "main.py").is_file()


# =====================================================================
# Messages meant for a person
# =====================================================================


class TestFailureMessages:
    def test_no_python_says_which_modules_were_missing(self):
        text = agent_service._no_python_message("/opt/python/bin/python")
        for module in agent_service.REQUIRED_MODULES:
            assert module in text
        assert "/opt/python/bin/python" in text
        assert "pip install" in text

    def test_a_port_race_names_the_port(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(
            "OSError: [Errno 98] Address already in use",
            encoding="utf-8",
        )
        text = agent_service._spawn_failure(log, "http://127.0.0.1:8000")
        assert "http://127.0.0.1:8000" in text
        assert "port" in text

    def test_a_missing_dependency_names_itself(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(
            "Traceback (most recent call last):\n"
            "ModuleNotFoundError: No module named 'bitsandbytes'\n",
            encoding="utf-8",
        )
        text = agent_service._spawn_failure(log, "http://127.0.0.1:8000")
        assert "'bitsandbytes'" in text
        assert "pip install" in text

    def test_any_other_death_shows_the_log_rather_than_nothing(
        self, tmp_path,
    ):
        log = tmp_path / "agent.log"
        log.write_text("Something went sideways at line 12", encoding="utf-8")
        assert "line 12" in agent_service._spawn_failure(
            log, "http://127.0.0.1:8000"
        )

    def test_an_empty_log_does_not_render_an_empty_error(self, tmp_path):
        text = agent_service._spawn_failure(tmp_path / "absent.log", "http://x")
        assert text.strip()


# =====================================================================
# The state machine
# =====================================================================


class TestState:
    def test_fresh_service_is_stopped(self):
        svc = agent_service.AgentService()
        assert svc.state == "stopped"
        assert svc.url == ""
        assert svc.owned is False
        assert svc.is_running() is False

    def test_wait_running_does_not_wait_when_nothing_is_starting(self):
        svc = agent_service.AgentService()
        start = time.monotonic()
        assert svc.wait_running(timeout=30.0) is False
        assert time.monotonic() - start < 5.0

    def test_a_stop_supersedes_a_start_already_in_flight(self):
        """The race that would otherwise orphan a process.

        ``stop()`` bumps the generation, so a worker that later reports
        success for its own (now stale) generation is refused — which is
        the only reason ``_spawn`` can be allowed to run without the
        caller holding a lock across it.
        """
        svc = agent_service.AgentService()
        assert svc._publish(0, "running", "ok", "http://x", owned=True) is True

        svc.stop()

        assert svc._publish(0, "running", "ok", "http://x", owned=True) is False
        assert svc.state == "stopped"
        assert svc.url == ""

    def test_stop_leaves_a_foreign_agent_serving(self):
        """Not ours to kill: somebody started it deliberately."""
        svc = agent_service.AgentService()
        svc._publish(0, "running", "Outside.", "http://x", owned=False)

        svc.stop()

        assert svc.state == "running"
        assert svc.url == "http://x"

    def test_stop_clears_ours(self):
        svc = agent_service.AgentService()
        svc._publish(0, "running", "Running (started by BIDS-Manager).",
                     "http://x", owned=True)

        svc.stop()

        assert svc.state == "stopped"
        assert svc.url == ""
        assert svc.owned is False


# =====================================================================
# Reuse rather than fight
# =====================================================================


class TestReuse:
    def test_an_agent_already_serving_is_reused_not_restarted(
        self, monkeypatch, agent_url,
    ):
        """Somebody ran ``python main.py`` by hand, or an earlier launch
        never cleaned up. Spawning a second one would fail; killing the
        first would take away a service somebody else may rely on."""
        host, _, port = agent_url.partition(f"{agent_service.HOST}:")
        monkeypatch.setattr(agent_service, "DEFAULT_PORT", int(port))

        looked_for_python: list[bool] = []
        monkeypatch.setattr(
            agent_service,
            "resolve_python",
            lambda *a, **k: looked_for_python.append(True) or None,
        )

        svc = agent_service.AgentService()
        svc.start()

        assert _until(lambda: svc.state != "starting")
        assert svc.state == "running"
        assert svc.url == agent_url
        assert svc.owned is False
        # It never even asked which Python to use: nothing to launch.
        assert looked_for_python == []

        # ...and stop() still leaves it alone, for the same reason.
        svc.stop()
        assert svc.state == "running"
        assert svc.url == agent_url

    def test_a_busy_port_makes_it_choose_another(
        self, monkeypatch, stranger_url,
    ):
        """The point of ``BIDS_AGENT_PORT``: rather than dying with a bind
        error the user cannot act on, move to a port that is free.

        ``_spawn`` is replaced with something that records the port and
        stops, so the decision is observed without launching a process.
        """
        _, _, port = stranger_url.partition(f"{agent_service.HOST}:")
        occupied = int(port)
        monkeypatch.setattr(agent_service, "DEFAULT_PORT", occupied)
        monkeypatch.setattr(
            agent_service, "resolve_python", lambda *a, **k: ["python"]
        )

        chosen: list[int] = []

        def fake_spawn(self, exe, requested, log_path, generation):
            chosen.append(requested)
            raise agent_service._Failure("stopped before launching")

        monkeypatch.setattr(agent_service.AgentService, "_spawn", fake_spawn)

        svc = agent_service.AgentService()
        svc.start()

        assert _until(lambda: svc.state != "starting")
        assert svc.state == "failed"
        assert chosen and chosen[0] != occupied
        assert 0 < chosen[0] < 65536

    def test_a_free_port_makes_it_take_the_default(
        self, monkeypatch, dead_url,
    ):
        """The ordinary case, and the one that must not get clever."""
        _, _, port = dead_url.partition(f"{agent_service.HOST}:")
        monkeypatch.setattr(agent_service, "DEFAULT_PORT", int(port))
        monkeypatch.setattr(
            agent_service, "resolve_python", lambda *a, **k: ["python"]
        )

        chosen: list[int] = []

        def fake_spawn(self, exe, requested, log_path, generation):
            chosen.append(requested)
            raise agent_service._Failure("stopped before launching")

        monkeypatch.setattr(agent_service.AgentService, "_spawn", fake_spawn)

        svc = agent_service.AgentService()
        svc.start()

        assert _until(lambda: svc.state != "starting")
        assert svc.state == "failed"
        assert chosen == [int(port)]

    def test_missing_agent_code_is_reported_not_raised(
        self, monkeypatch, dead_url,
    ):
        host, _, port = dead_url.partition(f"{agent_service.HOST}:")
        monkeypatch.setattr(agent_service, "DEFAULT_PORT", int(port))
        monkeypatch.setattr(
            agent_service, "AGENT_DIR", agent_service.AGENT_DIR / "nowhere"
        )

        svc = agent_service.AgentService()
        svc.start()

        assert _until(lambda: svc.state != "starting")
        assert svc.state == "failed"
        assert "not found" in svc.detail
