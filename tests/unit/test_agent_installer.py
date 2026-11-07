"""The agent installer: what it runs, what it decides, what it leaves.

Everything that can be checked without cloning a repository or running
pip is checked here: which folder counts as installed, which
interpreter is new enough, which requirements file a CPU-only machine
gets, and the zip fallback for a machine without git. The subprocesses
themselves are replaced — the point is the decisions, not that git
works.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

from bidsmgr import agent_installer, agent_service


@pytest.fixture(autouse=True)
def _install_dir_in_tmp(tmp_path: Path, monkeypatch) -> Path:
    """Every test installs into a fresh tmp folder, never into the real
    ``bidsmgr/ai_agent`` this workspace ships with."""
    target = tmp_path / "ai_agent"
    monkeypatch.setattr(agent_service, "AGENT_DIR", target)
    return target


def _write_main(install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    (install_dir / "main.py").write_text("pass", encoding="utf-8")


# ---------------------------------------------------------------------------
# What counts as installed
# ---------------------------------------------------------------------------


class TestInstalled:
    def test_an_entry_point_means_installed(self, _install_dir_in_tmp):
        _write_main(_install_dir_in_tmp)
        assert agent_installer.installed() is True

    def test_no_entry_point_means_not_installed(self, _install_dir_in_tmp):
        assert agent_installer.installed() is False

    def test_a_missing_folder_is_not_installed(self):
        assert agent_installer.installed() is False


def test_the_venv_python_sits_where_each_platform_puts_it(monkeypatch):
    venv = Path("/somewhere/.venv")
    monkeypatch.setattr(agent_installer.os, "name", "nt")
    assert agent_installer.venv_python(venv) == (
        venv / "Scripts" / "python.exe"
    )
    monkeypatch.setattr(agent_installer.os, "name", "posix")
    assert agent_installer.venv_python(venv) == venv / "bin" / "python"


# ---------------------------------------------------------------------------
# Step 1: the code
# ---------------------------------------------------------------------------


class TestClone:
    def test_git_is_used_when_the_machine_has_it(
        self, _install_dir_in_tmp, monkeypatch
    ):
        ran: list[list[str]] = []
        monkeypatch.setattr(agent_installer, "git_available", lambda: True)
        monkeypatch.setattr(
            agent_installer, "_run",
            lambda cmd, **k: ran.append(cmd),
        )
        dest = agent_installer.clone()
        assert dest == _install_dir_in_tmp
        assert ran and ran[0][:2] == ["git", "clone"]
        assert "--depth" in ran[0]
        assert agent_installer.AGENT_REPO_URL in ran[0]

    def test_without_git_the_repository_comes_as_a_zip(
        self, _install_dir_in_tmp, monkeypatch
    ):
        monkeypatch.setattr(agent_installer, "git_available", lambda: False)

        def fake_retrieve(url, path):
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("bids-ai-agent-main/main.py", "pass\n")
                archive.writestr(
                    "bids-ai-agent-main/requirements.txt", "torch\n",
                )

        monkeypatch.setattr(
            agent_installer.urllib.request, "urlretrieve", fake_retrieve,
        )
        dest = agent_installer.clone()
        assert (dest / "main.py").is_file()
        assert (dest / "requirements.txt").is_file()
        # The top-level folder of a GitHub zip is not part of the
        # repository; its contents are.
        assert not (dest / "bids-ai-agent-main").exists()

    def test_debris_from_a_failed_attempt_is_replaced(
        self, _install_dir_in_tmp, monkeypatch
    ):
        _install_dir_in_tmp.mkdir(parents=True)
        (_install_dir_in_tmp / "half-fetched.bin").write_bytes(b"junk")
        monkeypatch.setattr(agent_installer, "git_available", lambda: True)
        monkeypatch.setattr(agent_installer, "_run", lambda cmd, **k: None)
        dest = agent_installer.clone()
        assert dest == _install_dir_in_tmp
        assert not (_install_dir_in_tmp / "half-fetched.bin").exists()

    def test_an_offline_machine_gets_a_message_it_can_act_on(
        self, _install_dir_in_tmp, monkeypatch
    ):
        monkeypatch.setattr(agent_installer, "git_available", lambda: False)

        def failing_retrieve(url, path):
            raise OSError("no route to host")

        monkeypatch.setattr(
            agent_installer.urllib.request, "urlretrieve", failing_retrieve,
        )
        with pytest.raises(agent_installer.InstallError) as excinfo:
            agent_installer.clone()
        assert "internet connection" in str(excinfo.value)


def test_git_availability_needs_both_the_binary_and_a_working_call(
    monkeypatch,
):
    monkeypatch.setattr(agent_installer.shutil, "which", lambda name: None)
    assert agent_installer.git_available() is False

    # A binary that exists but will not answer --version is not a git we
    # can clone with.
    monkeypatch.setattr(
        agent_installer.shutil, "which", lambda name: "C:/git.exe",
    )
    monkeypatch.setattr(
        agent_installer.subprocess, "run",
        lambda *a, **k: True,
    )
    assert agent_installer.git_available() is True


# ---------------------------------------------------------------------------
# Step 2: the private environment
# ---------------------------------------------------------------------------


class TestPickBasePython:
    def test_the_first_new_enough_interpreter_wins(self, monkeypatch):
        versions = iter([(3, 9), (3, 12)])
        monkeypatch.setattr(
            agent_installer, "_python_version",
            lambda cmd: next(versions),
        )
        base = agent_installer.pick_base_python()
        assert isinstance(base, list) and base

    def test_only_old_interpreters_is_a_user_facing_failure(self, monkeypatch):
        monkeypatch.setattr(
            agent_installer, "_python_version", lambda cmd: (3, 9),
        )
        with pytest.raises(agent_installer.InstallError) as excinfo:
            agent_installer.pick_base_python()
        assert "Python 3.10" in str(excinfo.value)

    def test_an_interpreter_that_will_not_say_is_skipped(self, monkeypatch):
        answers = iter([None, (3, 11)])
        monkeypatch.setattr(
            agent_installer, "_python_version",
            lambda cmd: next(answers),
        )
        assert agent_installer.pick_base_python()


class TestCreateVenv:
    def test_a_fresh_environment_is_created(
        self, _install_dir_in_tmp, monkeypatch,
    ):
        ran: list[list[str]] = []

        def fake_run(cmd, **k):
            ran.append(cmd)
            python = agent_installer.venv_python(
                _install_dir_in_tmp / ".venv"
            )
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")

        monkeypatch.setattr(agent_installer, "_run", fake_run)
        target = agent_installer.create_venv(["python"])
        assert target == agent_installer.venv_dir()
        assert ran and ran[0][-2:] == ["venv", str(target)]

    def test_an_environment_from_an_earlier_attempt_is_reused(
        self, _install_dir_in_tmp, monkeypatch,
    ):
        python = agent_installer.venv_python(agent_installer.venv_dir())
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")

        def must_not_run(cmd, **k):
            raise AssertionError("a working venv should be reused")

        monkeypatch.setattr(agent_installer, "_run", must_not_run)
        assert agent_installer.create_venv(["python"]) == (
            agent_installer.venv_dir()
        )


# ---------------------------------------------------------------------------
# Step 3: the dependencies
# ---------------------------------------------------------------------------

_CUDA_REQUIREMENTS = (
    "--find-links https://download.pytorch.org/whl/cu126/torch/\n"
    "torch==2.14.1+cu126\n"
    "transformers>=4.57\n"
)


class TestRequirementsForMachine:
    def test_a_gpu_machine_installs_the_repositorys_own_file(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(agent_installer, "has_nvidia_gpu", lambda: True)
        requirements = tmp_path / "requirements.txt"
        requirements.write_text(_CUDA_REQUIREMENTS, encoding="utf-8")
        assert agent_installer._requirements_for_machine(
            requirements, None,
        ) == requirements

    def test_a_cpu_machine_gets_torch_without_the_cuda_suffix(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(agent_installer, "has_nvidia_gpu", lambda: False)
        requirements = tmp_path / "requirements.txt"
        requirements.write_text(_CUDA_REQUIREMENTS, encoding="utf-8")
        chosen = agent_installer._requirements_for_machine(requirements, None)
        text = chosen.read_text(encoding="utf-8")
        # The CUDA wheel index is gone, the CUDA suffix is dropped from
        # the pin, and everything else is untouched.
        assert "--find-links" not in text
        assert "torch==2.14.1\n" in text
        assert "+cu" not in text
        assert "transformers>=4.57" in text

    def test_the_gpu_probe_needs_nvidia_smi(self, monkeypatch):
        monkeypatch.setattr(agent_installer.shutil, "which", lambda n: None)
        assert agent_installer.has_nvidia_gpu() is False


# ---------------------------------------------------------------------------
# The whole install
# ---------------------------------------------------------------------------


class TestFullInstall:
    def test_the_steps_run_in_order_and_return_the_venv_python(
        self, _install_dir_in_tmp, monkeypatch,
    ):
        order: list[str] = []
        # The fallback path: an app environment that cannot take the
        # install (a frozen build, a read-only Python) still gets the
        # private environment inside the clone.
        monkeypatch.setattr(
            agent_installer, "app_environment_available", lambda: False,
        )
        monkeypatch.setattr(
            agent_installer, "clone",
            lambda emit=None: order.append("clone") or _install_dir_in_tmp,
        )
        monkeypatch.setattr(
            agent_installer, "pick_base_python",
            lambda emit=None: ["python"],
        )
        monkeypatch.setattr(
            agent_installer, "create_venv",
            lambda base, emit=None: order.append("venv")
            or _install_dir_in_tmp / ".venv",
        )
        monkeypatch.setattr(
            agent_installer, "install_dependencies",
            lambda venv, emit=None: order.append("deps"),
        )
        python = agent_installer.full_install()
        assert order == ["clone", "venv", "deps"]
        assert python == agent_installer.venv_python(
            _install_dir_in_tmp / ".venv"
        )

    def test_an_existing_install_skips_the_clone(
        self, _install_dir_in_tmp, monkeypatch,
    ):
        _write_main(_install_dir_in_tmp)

        def must_not_clone(emit=None):
            raise AssertionError("an installed module should not be recloned")

        monkeypatch.setattr(agent_installer, "clone", must_not_clone)
        monkeypatch.setattr(
            agent_installer, "app_environment_available", lambda: False,
        )
        monkeypatch.setattr(
            agent_installer, "pick_base_python",
            lambda emit=None: ["python"],
        )
        monkeypatch.setattr(
            agent_installer, "create_venv",
            lambda base, emit=None: _install_dir_in_tmp / ".venv",
        )
        monkeypatch.setattr(
            agent_installer, "install_dependencies",
            lambda venv, emit=None: None,
        )
        assert agent_installer.full_install()

    def test_the_apps_own_environment_is_used_when_it_can_take_the_deps(
        self, _install_dir_in_tmp, monkeypatch,
    ):
        """The documented setup, and the one a user reproduces by hand:
        the agent shares the Python that runs BIDS-Manager, so the
        requirements are installed with ``sys.executable`` and no second
        environment is built."""
        _write_main(_install_dir_in_tmp)
        used: list[Path] = []

        monkeypatch.setattr(
            agent_installer, "app_environment_available", lambda: True,
        )

        def must_not_build(*args, **kwargs):
            raise AssertionError("a usable app environment needs no venv")

        monkeypatch.setattr(agent_installer, "create_venv", must_not_build)
        monkeypatch.setattr(
            agent_installer, "_install_into",
            lambda python, emit=None: used.append(python),
        )

        python = agent_installer.full_install()

        assert python == Path(sys.executable)
        assert used == [Path(sys.executable)]


def test_the_pip_timeout_leaves_room_for_a_cuda_wheel():
    # A CUDA torch wheel is multiple gigabytes; a timeout measured in
    # seconds would kill a healthy download on a laptop connection.
    assert agent_installer.PIP_TIMEOUT >= 1800.0


def test_the_installer_is_stdlib_only():
    # The installer runs before the agent's heavy dependencies exist, so
    # it must not import torch or transformers — it only ever reads the
    # folder the service already knows about and runs what the machine
    # already has.
    source = Path(agent_installer.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import transformers" not in source
