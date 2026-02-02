# Project: EdgeTPUOnModernRPIs

Client library + systemd service for running Coral Edge TPU inference on Raspberry Pi OS Trixie, where system Python 3.13 is too new for tflite-runtime.

## Architecture (two-process design)

- **Service** (`service.py`): Runs in a Python 3.11 venv (via pyenv) with tflite-runtime. Listens on `/tmp/edgetpu.sock`. Manages TPU slots, model affinity, LRU eviction, per-TPU worker threads.
- **Client** (`client.py`): Runs on system Python 3.13. Communicates with service over Unix socket using binary numpy protocol. Installed via `pip install`.
- **CLI** (`cli.py`): Thin wrapper around client for shell/LLM tool use.

## Critical: Dual-file sync for service.py

`service.py` exists in TWO places that must stay in sync:

1. `src/rpi_edgetpu/service.py` — **source of truth**
2. `install.sh` — embedded as a heredoc inside `write_service_py()` between `SERVICEEOF` markers

**After ANY change to `service.py`, you MUST run:**
```bash
python3 scripts/sync_service.py
```

This replaces the heredoc block in `install.sh` with the current `service.py` contents. Always commit both files together.

## Repository layout

```
src/rpi_edgetpu/
  service.py     ← source of truth for Edge TPU service
  client.py      ← client library (EdgeTPUClient, EdgeTPUError, EdgeTPUBusyError)
  cli.py         ← edgetpu-cli entry point
  __init__.py    ← exports EdgeTPUClient, EdgeTPUError, EdgeTPUBusyError
  templates/     ← template files
scripts/
  sync_service.py  ← keeps install.sh heredoc in sync with service.py
install.sh       ← standalone curl-able installer (contains embedded service.py copy)
examples/        ← annotated usage scripts (run on RPi with service running)
pyproject.toml   ← package config, entry point: edgetpu-cli
```

## Download URLs

Binary dependencies (tflite-runtime, pycoral, libedgetpu) are hosted on our own GitHub release at `2.17.1`, not upstream feranick repos. The URLs are defined at the top of `install.sh` (lines ~20-22). Originally from feranick's builds — credit is preserved in user-facing messages.

## Wire protocol

Client-service communication uses a binary protocol over Unix socket:
- Header: `[4-byte header length][JSON header][optional numpy bytes]`
- Commands: `load_model`, `infer`, `detect`, `embedding`, `ping`, `rescan_tpus`, `quit`
- Responses for `infer`/`embedding`: `[4-byte array size][header][numpy bytes]` (size=0 means JSON error)
- Other responses: `[4-byte length][JSON]`

Changes to the protocol must be coordinated between `service.py` and `client.py`.

## Common pitfalls

- **load_model responses**: Both `loaded` and `already_loaded` statuses must return the same metadata fields (`input_shape`, `input_dtype`, `output_shape`). Example scripts and client code depend on these fields being present regardless of status.
- **install.sh is not just shell**: It contains a full Python service (~560 lines) as a heredoc. Don't edit the embedded Python directly — edit `service.py` and sync.
- **No test suite**: Testing happens manually on a Raspberry Pi 4 with Coral USB Accelerator(s). Example scripts in `examples/` serve as integration tests.

## Commit messages

- Use imperative mood, focus on "why" not "what"
- Follow existing style: short subject line, optional body with context

## Maintaining this file

Keep this CLAUDE.md updated as you work on the project. When you discover new pitfalls, learn about undocumented conventions, fix non-obvious bugs, or encounter anything that would save time in a future session, add it here.
