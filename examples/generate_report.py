#!/usr/bin/env python3
"""
EdgeTPU diagnostic and integration test report generator.

Collects system/installation diagnostics and runs a full integration test suite
exercising the service, Python client, CLI, and multi-TPU routing. Output is a
plain-text report suitable for copy-pasting into a GitHub issue.

Usage:
    python generate_report.py                  # print + save to edgetpu_report.txt
    python generate_report.py -o report.txt    # print + save to custom path
"""

import argparse
import datetime
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from collections import namedtuple

MODEL_URL = (
    "https://github.com/ricardodeazambuja/EdgeTPUModelZoo/raw/"
    "refs/heads/master/mobilenet_v2_1.0_224_quant/"
    "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
)
MODEL_PATH = "/tmp/edgetpu_test_model.tflite"
SOCKET_PATH = "/tmp/edgetpu.sock"

CheckResult = namedtuple("CheckResult", ["name", "passed", "detail"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(args, timeout=10):
    """Run a command and return (rc, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", "timed out"
    except Exception as e:
        return -1, "", str(e)


def redact_home(text):
    """Replace /home/<username> with ~ for privacy."""
    home = pathlib.Path.home()
    return text.replace(str(home), "~")


def download_model(url, dest):
    """Download model file; skip if already cached. Returns (ok, detail)."""
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"Using cached model ({dest.stat().st_size} bytes)"
    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size
        if size == 0:
            return False, "Downloaded file is empty"
        return True, f"Downloaded model ({size} bytes)"
    except Exception as e:
        return False, f"Download failed: {e}"


def fmt_lines(text, indent="  "):
    """Indent every line of text."""
    return "\n".join(f"{indent}{line}" for line in text.splitlines())


# ---------------------------------------------------------------------------
# System checks (1-12)
# ---------------------------------------------------------------------------

def check_os():
    lines = []
    # /etc/os-release
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith(("PRETTY_NAME=", "VERSION_ID=", "ID=")):
                    lines.append(line.strip())
    except FileNotFoundError:
        lines.append("(no /etc/os-release)")

    lines.append(f"Arch: {platform.machine()}")
    lines.append(f"Kernel: {platform.release()}")
    lines.append(f"Hostname: {platform.node()}")
    return CheckResult("OS / Kernel", True, "\n".join(lines))


def check_system_python():
    ver = sys.version_info
    passed = ver >= (3, 9)
    detail = (
        f"Python {sys.version}\n"
        f"Executable: {redact_home(sys.executable)}"
    )
    if not passed:
        detail += "\n(requires >= 3.9)"
    return CheckResult("System Python", passed, detail)


def check_rpi_edgetpu_package():
    lines = []
    passed = True
    try:
        import rpi_edgetpu
        lines.append(f"rpi_edgetpu {rpi_edgetpu.__version__}")
    except ImportError as e:
        lines.append(f"rpi_edgetpu: IMPORT FAILED ({e})")
        passed = False

    try:
        import numpy as np
        lines.append(f"numpy {np.__version__}")
    except ImportError as e:
        lines.append(f"numpy: IMPORT FAILED ({e})")
        passed = False

    return CheckResult("rpi-edgetpu package", passed, "\n".join(lines))


def check_pyenv():
    lines = []
    pyenv_root = os.environ.get("PYENV_ROOT", "")
    if not pyenv_root:
        pyenv_root = str(pathlib.Path.home() / ".pyenv")

    exists = pathlib.Path(pyenv_root).is_dir()
    lines.append(f"PYENV_ROOT: {redact_home(pyenv_root)} ({'exists' if exists else 'NOT FOUND'})")

    rc, out, _ = run_cmd(["pyenv", "--version"])
    if rc == 0:
        lines.append(f"pyenv binary: {out.strip()}")
    else:
        lines.append("pyenv binary: NOT FOUND")

    passed = exists and rc == 0
    return CheckResult("pyenv", passed, "\n".join(lines))


def check_coral_venv():
    lines = []
    venv_dir = os.environ.get("CORAL_VENV", "")
    if not venv_dir:
        venv_dir = str(pathlib.Path.home() / ".coral-venv")

    venv_path = pathlib.Path(venv_dir)
    exists = venv_path.is_dir()
    lines.append(f"Venv dir: {redact_home(venv_dir)} ({'exists' if exists else 'NOT FOUND'})")

    if not exists:
        return CheckResult("Coral venv", False, "\n".join(lines))

    venv_python = venv_path / "bin" / "python"
    rc, out, _ = run_cmd([str(venv_python), "--version"])
    if rc == 0:
        lines.append(f"Venv Python: {out.strip()}")
    else:
        lines.append("Venv Python: FAILED to run")
        return CheckResult("Coral venv", False, "\n".join(lines))

    rc, out, _ = run_cmd([str(venv_path / "bin" / "pip"), "freeze"])
    if rc == 0:
        for pkg in ("tflite-runtime", "pycoral", "numpy"):
            match = [l for l in out.splitlines() if l.lower().startswith(pkg)]
            if match:
                lines.append(f"  {match[0]}")
            else:
                lines.append(f"  {pkg}: NOT INSTALLED")
    else:
        lines.append("pip freeze: FAILED")

    return CheckResult("Coral venv", True, "\n".join(lines))


def check_libedgetpu():
    rc, out, err = run_cmd(["dpkg", "-s", "libedgetpu1-std"])
    if rc == 0:
        version_lines = [l for l in out.splitlines() if l.startswith("Version:")]
        detail = version_lines[0] if version_lines else "installed"
        return CheckResult("libedgetpu", True, detail)
    return CheckResult("libedgetpu", False, "libedgetpu1-std not installed")


def check_usb_devices():
    rc, out, _ = run_cmd(["lsusb"])
    if rc != 0:
        return CheckResult("USB TPU devices", False, "lsusb command failed")

    tpu_lines = [
        l for l in out.splitlines()
        if "1a6e" in l.lower() or "18d1" in l.lower()
    ]
    if tpu_lines:
        detail = f"Found {len(tpu_lines)} TPU device(s):\n" + "\n".join(f"  {l.strip()}" for l in tpu_lines)
        return CheckResult("USB TPU devices", True, detail)
    return CheckResult("USB TPU devices", False, "No Coral USB devices found (vendor 1a6e/18d1)")


def check_service_script():
    candidates = []
    service_dir = os.environ.get("CORAL_SERVICE_DIR", "")
    if service_dir:
        candidates.append(pathlib.Path(service_dir) / "edgetpu_service.py")
    candidates.append(pathlib.Path.home() / "coral-service" / "edgetpu_service.py")

    for p in candidates:
        if p.exists():
            detail = f"Found: {redact_home(str(p))} ({p.stat().st_size} bytes)"
            return CheckResult("Service script", True, detail)

    searched = ", ".join(redact_home(str(p.parent)) for p in candidates)
    return CheckResult("Service script", False, f"Not found in: {searched}")


def check_systemd_service():
    lines = []
    rc_active, out_active, _ = run_cmd(["systemctl", "is-active", "edgetpu"])
    rc_enabled, out_enabled, _ = run_cmd(["systemctl", "is-enabled", "edgetpu"])
    is_active = out_active.strip() == "active"
    lines.append(f"is-active: {out_active.strip()}")
    lines.append(f"is-enabled: {out_enabled.strip()}")

    # Properties
    rc, out, _ = run_cmd([
        "systemctl", "show", "edgetpu",
        "--property=MainPID,MemoryCurrent,ActiveEnterTimestamp"
    ])
    if rc == 0 and out.strip():
        for prop_line in out.strip().splitlines():
            lines.append(f"  {prop_line}")

    if not is_active:
        lines.append("\nLast 20 journal lines:")
        rc, out, _ = run_cmd(["journalctl", "-u", "edgetpu", "-n", "20", "--no-pager"])
        if rc == 0:
            lines.append(redact_home(out.strip()))
        else:
            lines.append("  (could not read journal)")

    return CheckResult("Systemd service", is_active, "\n".join(lines))


def check_socket():
    sock_path = pathlib.Path(SOCKET_PATH)
    if not sock_path.exists():
        return CheckResult("Socket file", False, f"{SOCKET_PATH} does not exist")

    try:
        st = sock_path.stat()
        is_socket = stat.S_ISSOCK(st.st_mode)
        perms = oct(st.st_mode)[-3:]
        detail = f"{SOCKET_PATH} (socket={is_socket}, perms={perms})"
        return CheckResult("Socket file", is_socket, detail)
    except Exception as e:
        return CheckResult("Socket file", False, str(e))


def check_model_download():
    ok, detail = download_model(MODEL_URL, MODEL_PATH)
    return CheckResult("Test model download", ok, detail)


def check_cli_available():
    path = shutil.which("edgetpu-cli")
    if path:
        return CheckResult("edgetpu-cli available", True, f"Found: {redact_home(path)}")
    # Try module invocation
    rc, out, _ = run_cmd([sys.executable, "-m", "rpi_edgetpu.cli", "--help"])
    if rc == 0:
        return CheckResult("edgetpu-cli available", True, f"Available via: {sys.executable} -m rpi_edgetpu.cli")
    return CheckResult("edgetpu-cli available", False, "Not found on PATH and module invocation failed")


# ---------------------------------------------------------------------------
# Integration tests (13-20+)
# ---------------------------------------------------------------------------

def _service_prereq(results):
    """Check whether service is running (from earlier results)."""
    for r in results:
        if r.name == "Systemd service" and not r.passed:
            return "Service is not running"
        if r.name == "Socket file" and not r.passed:
            return "Socket file not found"
    return None


def _model_prereq(results):
    """Check whether model download succeeded."""
    for r in results:
        if r.name == "Test model download" and not r.passed:
            return "Test model download failed"
    return None


def test_client_ping(results):
    skip = _service_prereq(results)
    if skip:
        return CheckResult("Python client: ping", False, f"SKIPPED — {skip}")
    try:
        from rpi_edgetpu import EdgeTPUClient
        t0 = time.monotonic()
        client = EdgeTPUClient()
        resp = client.ping()
        rtt_ms = (time.monotonic() - t0) * 1000
        client.close()
        ok = isinstance(resp, dict) and resp.get("status") == "ok"
        version = resp.get("version", "N/A")
        detail = f"Response: {resp}\nService version: {version}\nRTT: {rtt_ms:.1f} ms"
        return CheckResult("Python client: ping", ok, detail)
    except Exception as e:
        return CheckResult("Python client: ping", False, str(e))


def test_client_load_model(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("Python client: load model", False, f"SKIPPED — {skip}")
    try:
        from rpi_edgetpu import EdgeTPUClient
        client = EdgeTPUClient()
        resp = client.load_model(MODEL_PATH)
        ok = isinstance(resp, dict) and resp.get("status") in ("loaded", "already_loaded")
        lines = [f"Status: {resp.get('status')}"]
        for key in ("input_shape", "input_dtype", "output_shape"):
            lines.append(f"  {key}: {resp.get(key)}")
        client.close()
        return CheckResult("Python client: load model", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("Python client: load model", False, str(e))


def test_client_inference(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("Python client: inference", False, f"SKIPPED — {skip}")
    try:
        import numpy as np
        from rpi_edgetpu import EdgeTPUClient

        client = EdgeTPUClient()
        meta = client.load_model(MODEL_PATH)
        input_shape = meta["input_shape"]
        input_dtype = meta["input_dtype"]

        # Resolve dtype string to numpy dtype
        dt = np.dtype(input_dtype)
        dummy = np.zeros(input_shape, dtype=dt)

        t0 = time.monotonic()
        output = client.infer(dummy)
        latency_ms = (time.monotonic() - t0) * 1000

        lines = [
            f"Output shape: {list(output.shape)}",
            f"Output dtype: {output.dtype}",
            f"Inference latency: {latency_ms:.1f} ms",
        ]

        expected_shape = meta.get("output_shape")
        shape_ok = list(output.shape) == expected_shape if expected_shape else True

        # Top-5 classes
        flat = output.flatten()
        top5_idx = flat.argsort()[-5:][::-1]
        lines.append("Top-5 classes (index: value):")
        for idx in top5_idx:
            lines.append(f"  {idx}: {flat[idx]}")

        client.close()
        ok = isinstance(output, np.ndarray) and shape_ok
        if not shape_ok:
            lines.append(f"Shape mismatch: expected {expected_shape}, got {list(output.shape)}")
        return CheckResult("Python client: inference", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("Python client: inference", False, str(e))


def test_client_pipeline(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("Python client: pipeline", False, f"SKIPPED — {skip}")
    try:
        import numpy as np
        from rpi_edgetpu import EdgeTPUClient

        client = EdgeTPUClient()
        meta = client.load_model(MODEL_PATH)
        input_shape = meta["input_shape"]
        input_dtype = meta["input_dtype"]
        dt = np.dtype(input_dtype)
        dummy = np.zeros(input_shape, dtype=dt)

        # Chain the same model twice — stage 1 will fail because the
        # output shape [1, 1001] doesn't match input shape [1, 224, 224, 3]
        try:
            client.pipeline([MODEL_PATH, MODEL_PATH], dummy)
            # If it somehow succeeds, that's unexpected but not a failure
            client.close()
            return CheckResult(
                "Python client: pipeline", True,
                "Pipeline succeeded (unexpected — model may accept its own output)"
            )
        except Exception as e:
            msg = str(e)
            client.close()
            # We expect a stage-failure error mentioning stage 1
            ok = "stage" in msg.lower() or "pipeline" in msg.lower()
            detail = f"Expected error at stage 1: {msg}"
            return CheckResult("Python client: pipeline", ok, detail)
    except Exception as e:
        return CheckResult("Python client: pipeline", False, str(e))


def test_client_already_loaded(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("Python client: already_loaded", False, f"SKIPPED — {skip}")
    try:
        from rpi_edgetpu import EdgeTPUClient

        client = EdgeTPUClient()
        meta1 = client.load_model(MODEL_PATH)
        meta2 = client.load_model(MODEL_PATH)
        client.close()

        ok = meta2.get("status") == "already_loaded"
        lines = [f"First load status: {meta1.get('status')}"]
        lines.append(f"Second load status: {meta2.get('status')}")

        # Check metadata fields match
        match = True
        for key in ("input_shape", "input_dtype", "output_shape"):
            v1, v2 = meta1.get(key), meta2.get(key)
            same = v1 == v2
            if not same:
                match = False
            lines.append(f"  {key}: {'match' if same else 'MISMATCH'} ({v2})")

        ok = ok and match
        return CheckResult("Python client: already_loaded", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("Python client: already_loaded", False, str(e))


def test_client_rescan(results):
    skip = _service_prereq(results)
    if skip:
        return CheckResult("Python client: rescan TPUs", False, f"SKIPPED — {skip}")
    try:
        from rpi_edgetpu import EdgeTPUClient
        client = EdgeTPUClient()
        resp = client.rescan_tpus()
        client.close()
        tpu_count = resp.get("tpu_count", 0)
        ok = isinstance(resp, dict) and tpu_count >= 1
        return CheckResult("Python client: rescan TPUs", ok, f"Response: {resp}")
    except Exception as e:
        return CheckResult("Python client: rescan TPUs", False, str(e))


def _cli_cmd():
    """Return the CLI command list prefix."""
    path = shutil.which("edgetpu-cli")
    if path:
        return [path]
    return [sys.executable, "-m", "rpi_edgetpu.cli"]


def test_cli_ping(results):
    skip = _service_prereq(results)
    if skip:
        return CheckResult("CLI: ping", False, f"SKIPPED — {skip}")
    try:
        cmd = _cli_cmd() + ["--json", "ping"]
        rc, out, err = run_cmd(cmd)
        lines = [f"Command: {' '.join(cmd)}", f"Exit code: {rc}"]
        if out.strip():
            parsed = json.loads(out)
            lines.append(f"Response: {parsed}")
            ok = rc == 0 and parsed.get("status") == "ok"
        else:
            lines.append(f"stderr: {err.strip()}")
            ok = False
        return CheckResult("CLI: ping", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("CLI: ping", False, str(e))


def test_cli_load_model(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("CLI: load-model", False, f"SKIPPED — {skip}")
    try:
        cmd = _cli_cmd() + ["--json", "load-model", MODEL_PATH]
        rc, out, err = run_cmd(cmd)
        lines = [f"Command: {redact_home(' '.join(cmd))}", f"Exit code: {rc}"]
        if out.strip():
            parsed = json.loads(out)
            lines.append(f"Response: {parsed}")
            ok = rc == 0 and "status" in parsed
        else:
            lines.append(f"stderr: {err.strip()}")
            ok = False
        return CheckResult("CLI: load-model", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("CLI: load-model", False, str(e))


def test_cli_infer(results):
    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("CLI: infer", False, f"SKIPPED — {skip}")
    try:
        import numpy as np

        # We need to know model input shape/dtype — load it first
        from rpi_edgetpu import EdgeTPUClient
        client = EdgeTPUClient()
        meta = client.load_model(MODEL_PATH)
        client.close()

        input_shape = meta["input_shape"]
        input_dtype = meta["input_dtype"]
        dt = np.dtype(input_dtype)

        dummy = np.zeros(input_shape, dtype=dt)
        input_path = "/tmp/edgetpu_test_input.npy"
        output_path = "/tmp/edgetpu_test_output.npy"
        np.save(input_path, dummy)

        # Clean up any previous output
        if os.path.exists(output_path):
            os.remove(output_path)

        cmd = _cli_cmd() + ["--json", "infer", MODEL_PATH, input_path, "-o", output_path]
        rc, out, err = run_cmd(cmd, timeout=15)
        lines = [f"Command: {redact_home(' '.join(cmd))}", f"Exit code: {rc}"]

        if out.strip():
            lines.append(f"stdout: {out.strip()}")
        if err.strip():
            lines.append(f"stderr: {err.strip()}")

        output_exists = os.path.exists(output_path)
        lines.append(f"Output file exists: {output_exists}")
        if output_exists:
            arr = np.load(output_path)
            lines.append(f"Output shape: {list(arr.shape)}, dtype: {arr.dtype}")

        ok = rc == 0 and output_exists
        return CheckResult("CLI: infer", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("CLI: infer", False, str(e))


def test_multi_tpu(results, tpu_count):
    if tpu_count < 2:
        return CheckResult(
            "Multi-TPU routing", None,
            f"SKIPPED — only {tpu_count} TPU(s) detected (need >= 2)"
        )

    skip = _service_prereq(results) or _model_prereq(results)
    if skip:
        return CheckResult("Multi-TPU routing", False, f"SKIPPED — {skip}")

    try:
        import numpy as np
        from rpi_edgetpu import EdgeTPUClient

        client_a = EdgeTPUClient()
        client_b = EdgeTPUClient()

        meta_a = client_a.load_model(MODEL_PATH)
        meta_b = client_b.load_model(MODEL_PATH)

        input_shape = meta_a["input_shape"]
        input_dtype = meta_a["input_dtype"]
        dt = np.dtype(input_dtype)
        dummy = np.zeros(input_shape, dtype=dt)

        out_a = client_a.infer(dummy)
        out_b = client_b.infer(dummy)

        lines = [
            f"Client A load status: {meta_a.get('status')}",
            f"Client B load status: {meta_b.get('status')}",
            f"Client A tpu_id: {meta_a.get('tpu_id', 'N/A')}",
            f"Client B tpu_id: {meta_b.get('tpu_id', 'N/A')}",
            f"Client A output shape: {list(out_a.shape)}",
            f"Client B output shape: {list(out_b.shape)}",
        ]

        client_a.close()
        client_b.close()

        ok = isinstance(out_a, np.ndarray) and isinstance(out_b, np.ndarray)
        return CheckResult("Multi-TPU routing", ok, "\n".join(lines))
    except Exception as e:
        return CheckResult("Multi-TPU routing", False, str(e))


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _status_str(r):
    if r.passed is None:
        return "[SKIP]"
    return "[PASS]" if r.passed else "[FAIL]"


def _run_check(fn, *args, label=None):
    """Run a check/test function, printing progress to stderr."""
    name = label or fn.__name__.replace("_", " ").strip()
    print(f"  Running: {name} ...", end="", flush=True, file=sys.stderr)
    result = fn(*args)
    print(f" {_status_str(result)}", file=sys.stderr)
    return result


def generate_report():
    """Run all checks and tests, return the report as a string."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Get package version
    pkg_version = "unknown"
    try:
        import rpi_edgetpu
        pkg_version = rpi_edgetpu.__version__
    except Exception:
        pass

    # Get service version from ping
    svc_version = "N/A"
    try:
        from rpi_edgetpu import EdgeTPUClient
        client = EdgeTPUClient()
        resp = client.ping()
        client.close()
        if isinstance(resp, dict):
            svc_version = resp.get("version", "N/A")
    except Exception:
        pass

    # --- Part 1: System checks ---
    print("Running system checks ...", file=sys.stderr)
    system_checks = [
        _run_check(check_os, label="OS / Kernel"),
        _run_check(check_system_python, label="System Python"),
        _run_check(check_rpi_edgetpu_package, label="rpi-edgetpu package"),
        _run_check(check_pyenv, label="pyenv"),
        _run_check(check_coral_venv, label="Coral venv"),
        _run_check(check_libedgetpu, label="libedgetpu"),
        _run_check(check_usb_devices, label="USB TPU devices"),
        _run_check(check_service_script, label="Service script"),
        _run_check(check_systemd_service, label="Systemd service"),
        _run_check(check_socket, label="Socket file"),
        _run_check(check_model_download, label="Test model download"),
        _run_check(check_cli_available, label="edgetpu-cli available"),
    ]

    # --- Part 2: Integration tests ---
    print("Running integration tests ...", file=sys.stderr)
    integration_tests = []
    integration_tests.append(_run_check(test_client_ping, system_checks, label="Python client: ping"))
    integration_tests.append(_run_check(test_client_load_model, system_checks, label="Python client: load model"))
    integration_tests.append(_run_check(test_client_inference, system_checks, label="Python client: inference"))
    integration_tests.append(_run_check(test_client_pipeline, system_checks, label="Python client: pipeline"))
    integration_tests.append(_run_check(test_client_already_loaded, system_checks, label="Python client: already_loaded"))
    rescan_result = _run_check(test_client_rescan, system_checks, label="Python client: rescan TPUs")
    integration_tests.append(rescan_result)
    integration_tests.append(_run_check(test_cli_ping, system_checks, label="CLI: ping"))
    integration_tests.append(_run_check(test_cli_load_model, system_checks, label="CLI: load-model"))
    integration_tests.append(_run_check(test_cli_infer, system_checks, label="CLI: infer"))

    # Multi-TPU (conditional)
    tpu_count = 0
    if rescan_result.passed:
        try:
            # Parse tpu_count from the rescan detail
            import re
            m = re.search(r"'tpu_count':\s*(\d+)", rescan_result.detail)
            if m:
                tpu_count = int(m.group(1))
        except Exception:
            pass
    multi_result = _run_check(test_multi_tpu, system_checks, tpu_count, label="Multi-TPU routing")
    integration_tests.append(multi_result)

    # --- Build report ---
    all_results = system_checks + integration_tests
    sep = "=" * 62

    lines = [
        sep,
        "  EdgeTPU Diagnostic + Test Report",
        f"  Generated: {now}",
        f"  rpi-edgetpu version: {pkg_version}",
        f"  service version: {svc_version}",
        sep,
        "",
    ]

    # Summary
    lines.append("--- Summary " + "-" * 50)

    lines.append("  System Checks:")
    for r in system_checks:
        lines.append(f"    {_status_str(r)} {r.name}")

    lines.append("  Integration Tests:")
    for r in integration_tests:
        lines.append(f"    {_status_str(r)} {r.name}")

    counted = [r for r in all_results if r.passed is not None]
    passed_count = sum(1 for r in counted if r.passed)
    lines.append("-" * 62)
    lines.append(f"  Result: {passed_count}/{len(counted)} checks passed")
    lines.append("")

    # Detailed sections
    for i, r in enumerate(all_results, 1):
        lines.append(f"--- {i}. {r.name} {'-' * max(1, 52 - len(r.name) - len(str(i)))}")
        lines.append(f"  Status: {_status_str(r)}")
        if r.detail:
            lines.append(fmt_lines(redact_home(r.detail)))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an EdgeTPU diagnostic and integration test report."
    )
    parser.add_argument(
        "-o", "--output",
        help="Save report to this file (default: edgetpu_report.txt in current directory)",
    )
    args = parser.parse_args()

    print("", file=sys.stderr)
    report = generate_report()

    output_path = args.output or "edgetpu_report.txt"
    with open(output_path, "w") as f:
        f.write(report)
        f.write("\n")

    print("", file=sys.stderr)
    print(report)
    print(f"\nReport saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
