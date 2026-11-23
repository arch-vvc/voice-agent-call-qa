"""The `python -m callqa` command line. No test runs the real `claude` CLI or touches the network."""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from callqa.__main__ import bind_server, main, parse_args, resolve_config

ROOT = Path(__file__).resolve().parents[1]  # the call QA folder, wherever it lives
# The monorepo root has the project CLAUDE.md. A standalone export of this folder has none.
REPO_CLAUDE_MD = ROOT.parents[1] / "CLAUDE.md" if ROOT.parent.name == "demos" else None
SMOKE_LINE = "SMOKE OK v1 70% (14/20) v2 95% (19/20)"
CHECK_20 = ("CHECK OK backend=offline callers=20 v1 70% (14/20) v2 95% (19/20) "
            "v2 leaks: none; v2 failures: wants_manager")
CHECK_5 = "CHECK OK backend=offline callers=5 v1 20% (1/5) v2 80% (4/5) v2 leaks: none; v2 failures: wants_manager"


def run_cli(*args, env=None, path=None):
    """Run the CLI in a subprocess. `env` entries replace the inherited ones; None removes a key."""
    full = dict(os.environ)
    full.pop("CALLQA_BACKEND", None)
    full.pop("ANTHROPIC_API_KEY", None)
    for key, value in (env or {}).items():
        if value is None:
            full.pop(key, None)
        else:
            full[key] = value
    if path is not None:
        full["PATH"] = path
    return subprocess.run([sys.executable, "-m", "callqa", *args], cwd=ROOT, env=full,
                          capture_output=True, text=True, timeout=60)


def test_help_exits_0():
    proc = run_cli("--help")
    assert proc.returncode == 0
    for flag in ("--backend", "--offline", "--callers", "--max-turns", "--workers", "--port",
                 "--no-browser", "--smoke", "--check", "--agent"):
        assert flag in proc.stdout


def test_smoke_prints_exact_line_ignoring_env():
    proc = run_cli("--smoke", env={"CALLQA_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "sk-dummy-never-used"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == SMOKE_LINE


def test_check_offline_twenty_prints_exact_line():
    proc = run_cli("--check", "--backend", "offline")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == CHECK_20


def test_check_offline_five_prints_exact_line():
    proc = run_cli("--check", "--offline", "--callers", "5")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == CHECK_5


def test_check_without_claude_degrades_and_warns(tmp_path):
    proc = run_cli("--check", "--callers", "5", path=str(tmp_path))  # an empty dir: no claude on PATH
    assert proc.returncode == 0, proc.stderr
    assert "claude CLI not found on PATH; running offline" in proc.stdout + proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == CHECK_5


def test_anthropic_without_key_exits_2():
    proc = run_cli("--backend", "anthropic", "--check")
    assert proc.returncode == 2
    assert "ANTHROPIC_API_KEY is not set" in proc.stderr


def fake_which(name):
    return f"/fake/bin/{name}"


def test_llm_backend_defaults_are_5_callers_4_turns(monkeypatch):
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)
    cfg = resolve_config(parse_args([]), which=fake_which)
    assert (cfg.backend.name, cfg.callers, cfg.max_turns, cfg.workers) == ("claude-cli", 5, 4, 2)


def test_offline_defaults_are_20_callers_8_turns(monkeypatch):
    monkeypatch.delenv("CALLQA_BACKEND", raising=False)
    cfg = resolve_config(parse_args(["--offline"]), which=fake_which)
    assert (cfg.backend.name, cfg.callers, cfg.max_turns) == ("offline", 20, 8)
    cfg = resolve_config(parse_args(["--offline", "--callers", "7", "--max-turns", "3", "--workers", "1"]))
    assert (cfg.callers, cfg.max_turns, cfg.workers) == (7, 3, 1)


@pytest.mark.parametrize("value", ["0", "-3"])
def test_callers_must_be_positive(value):
    proc = run_cli("--offline", "--check", "--callers", value)
    assert proc.returncode == 2
    assert "--callers must be at least 1" in proc.stderr


def test_busy_port_moves_to_next():
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        busy = blocker.getsockname()[1]
        cfg = resolve_config(parse_args(["--offline", "--port", str(busy)]))
        server = bind_server(cfg, busy)
        try:
            assert busy < server.server_address[1] <= busy + 10
        finally:
            server.server_close()


def test_main_check_aborted_message(monkeypatch, capsys):
    import callqa.__main__ as cli
    from evals.run_evals import RunAborted

    def boom(*args, **kwargs):
        raise RunAborted("usage limit reached, try later or use --backend offline", [])

    monkeypatch.setattr(cli, "run_check", boom)
    assert main(["--offline", "--check"]) == 2
    assert "ABORTED: usage limit reached" in capsys.readouterr().out




def slow_respond(history):
    """A stand in agent for the Ctrl-C test: one second per reply."""
    time.sleep(1)
    return "Namaste, this is Asha from Acme Finance."


def start_server(*extra):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if k not in ("CALLQA_BACKEND", "ANTHROPIC_API_KEY")}
    proc = subprocess.Popen([sys.executable, "-m", "callqa", "--offline", "--no-browser", "--port", str(port), *extra],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/api/config", timeout=2).read()
            return proc, base, port
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise AssertionError("server did not start")


def test_ctrl_c_mid_run_exits_promptly():
    proc, base, _ = start_server("--agent", "v1=slow=tests.test_cli:slow_respond", "--workers", "2")
    try:
        req = urllib.request.Request(base + "/api/run", data=json.dumps({"sop": "x", "agent": "v1"}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=6)  # without the fix the pool works through all 20 calls first
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def test_ctrl_c_when_idle_exits_cleanly():
    proc, _, _ = start_server()
    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=6)
    assert proc.returncode == 0


def test_printed_url_uses_127_0_0_1():
    proc, _, port = start_server()
    try:
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=6)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert f"http://127.0.0.1:{port}" in out
    assert "localhost" not in out


def test_open_browser_uses_127_0_0_1(monkeypatch):
    import callqa.__main__ as cli
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)

    class StopNow(threading.Thread):  # serve() returns at once instead of waiting for Ctrl-C
        def is_alive(self):
            return False

    monkeypatch.setattr(cli.threading, "Thread", StopNow)
    assert main(["--offline", "--port", "0"]) == 0
    assert opened and opened[0].startswith("http://127.0.0.1:")




def test_command_dispatch_leaves_flags_working():
    smoke = run_cli("--smoke")
    assert smoke.returncode == 0, smoke.stderr
    assert smoke.stdout.strip().splitlines()[-1] == SMOKE_LINE
    check = run_cli("--check", "--backend", "offline")
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip().splitlines()[-1] == CHECK_20
    check5 = run_cli("--check", "--backend", "offline", "--callers", "5")
    assert check5.stdout.strip().splitlines()[-1] == CHECK_5
    ran = run_cli("eval", "--agent", "v1", "--backend", "offline", "--no-save")
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines()[0] == "EVAL agent=v1 set=story callers=20 repeats=1 seed=0 backend=offline"


def test_dashboard_flags_set_and_repeats():
    args = parse_args(["--set", "full", "--repeats", "3", "--offline"])
    assert (args.set, args.repeats) == ("full", 3)
    cfg = resolve_config(args)
    assert (cfg.set_name, cfg.repeats, cfg.callers) == ("full", 3, 20)  # callers stays the story count
    assert parse_args([]).set == "story" and parse_args([]).repeats == 1
    for bad in (["--repeats", "6"], ["--repeats", "0"], ["--set", "huge"], ["--set", "full", "--callers", "5"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_dashboard_over_200_calls_on_an_llm_backend_needs_yes(capsys):
    fake_which = lambda name: "/usr/bin/claude"  # noqa: E731  (never executed: main stops before any run)
    from callqa import __main__ as cli
    big = parse_args(["--backend", "claude-cli", "--set", "full", "--repeats", "5"])
    cfg = resolve_config(big, which=fake_which)
    assert cli.guard_large_run(cfg, big) == 2
    assert "--yes" in capsys.readouterr().err
    assert cli.guard_large_run(cfg, parse_args(["--backend", "claude-cli", "--set", "full", "--repeats", "5",
                                                "--yes"])) == 0
    assert cli.guard_large_run(resolve_config(parse_args(["--offline", "--set", "full", "--repeats", "5"])), big) == 0


@pytest.fixture
def stub_claude(tmp_path):
    """A `claude` first on PATH that records any call. Returns (PATH, marker file)."""
    marker = tmp_path / "called"
    stub = tmp_path / "bin" / "claude"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
    stub.chmod(0o755)
    return f"{stub.parent}{os.pathsep}{os.environ['PATH']}", marker


def test_dashboard_over_200_calls_exits_2_without_yes_and_starts_with_it(stub_claude):
    path, marker = stub_claude
    proc = run_cli("--backend", "claude-cli", "--callers", "20", "--no-browser", path=path)
    assert proc.returncode == 2 and "--yes" in proc.stderr
    assert not marker.exists()
    env = {k: v for k, v in os.environ.items() if k not in ("CALLQA_BACKEND", "ANTHROPIC_API_KEY")}
    env["PATH"] = path
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "callqa", "--backend", "claude-cli", "--callers", "20", "--yes",
                               "--no-browser", "--port", str(port)], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert server.stdout.readline().startswith("Call QA dashboard: http://127.0.0.1:")
    finally:
        server.kill()
        server.wait()
        server.stdout.close()
        server.stderr.close()
    assert not marker.exists()


def test_check_on_an_llm_backend_over_200_calls_needs_yes(stub_claude):
    path, marker = stub_claude
    proc = run_cli("--check", "--backend", "claude-cli", "--callers", "20", path=path)  # v1 and v2: 440 calls
    assert proc.returncode == 2 and "--yes" in proc.stderr
    assert not marker.exists()
