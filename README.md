# rpi-edgetpu

Client library and CLI for running [Coral Edge TPU](https://coral.ai/) inference on Raspberry Pi OS Trixie, where the system Python (3.13) is too new for `tflite-runtime`.

## Architecture

```
┌─────────────────────────┐                       ┌──────────────────────────────┐
│  Client A (thread)      │──┐                    │  coral-venv Python 3.11      │
│  model: X               │  │   Unix Socket      │                              │
├─────────────────────────┤  ├───────────────────>│  TPUManager                  │
│  Client B (thread)      │──┘  /tmp/edgetpu.sock │    route(model) → slot       │
│  model: Y               │                       │                              │
└─────────────────────────┘                       │  TPU 0: Queue → Worker (X)   │
                                                  │  TPU 1: Queue → Worker (Y)   │
                                                  │  - tflite-runtime            │
                                                  └──────────────────────────────┘
```

Your application runs on the system Python (3.13 on Trixie) with system-installed packages like `picamera2` and `numpy`. The Edge TPU service runs in a separate Python 3.11 venv (via pyenv) where `tflite-runtime` is supported. They communicate over a Unix socket using a binary numpy protocol.

The service automatically detects all connected Coral USB Accelerators at startup. Each TPU gets its own worker thread and queue. When a client loads a model, the **TPUManager** routes it using model affinity: if a TPU already has that model loaded it reuses it, otherwise it picks an empty TPU, or evicts the least-recently-used model. Multiple clients using the same model share a single TPU with no reloading. Clients using different models are routed to different TPUs when available.

Each client connection tracks its own model binding (per-client local state), so one client's `load_model` never silently affects another client's inference. On a single TPU with multiple models, the service detects eviction and automatically reloads the correct model before inference.

If the queue is full, clients receive an `EdgeTPUBusyError` they can catch and retry. New TPUs can be discovered at runtime via the `rescan_tpus()` client method or `edgetpu-cli rescan-tpus` command (hot-plug support).

## Service Setup

Install the Edge TPU inference service with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUOnModernRPIs/main/install.sh | bash
```

The installer is interactive — it walks through 6 steps (apt dependencies, pyenv, Python 3.11, venv, service script, systemd unit) and asks for confirmation before each one. Steps that are already complete are skipped automatically.

To check status, update, or uninstall:

```bash
# Download and run with status/update/uninstall argument
curl -fsSL https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUOnModernRPIs/main/install.sh | bash -s status
curl -fsSL https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUOnModernRPIs/main/install.sh | bash -s update
curl -fsSL https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUOnModernRPIs/main/install.sh | bash -s uninstall
```

Or if you have the script locally:

```bash
bash install.sh status
bash install.sh update
bash install.sh uninstall
```

## Client Installation

The client library (`rpi_edgetpu`) lets your Python code and the `edgetpu-cli` command talk to the running service. Install it on the same Pi where the service runs.

### Option A: venv with system site packages (recommended)

```bash
python3 -m venv --system-site-packages ~/my-project-venv
source ~/my-project-venv/bin/activate
pip install git+https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs.git
```

This gets `picamera2`, `numpy`, and `libcamera` from system packages while keeping your project dependencies isolated.

### Option B: Install into system Python

```bash
pip install --break-system-packages git+https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs.git
```

Simpler, but modifies the system Python environment.

### What gets installed (both options)

- `rpi_edgetpu` Python package (`EdgeTPUClient`, `EdgeTPUBusyError`)
- `edgetpu-cli` command-line tool
- Dependency: `numpy` (already present on Trixie)

### Upgrading

```bash
# Upgrade client library only (no dependency changes)
pip install --upgrade --no-deps git+https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs.git

# Update service script on the Pi (re-deploys and restarts systemd)
curl -fsSL https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUOnModernRPIs/main/install.sh | bash -s update
# Or locally:
bash install.sh update
```

## Client Usage

### Python API

```python
import numpy as np
from rpi_edgetpu import EdgeTPUClient, EdgeTPUBusyError

# Connect to the running service
client = EdgeTPUClient()

# Load a model
info = client.load_model("/path/to/model_edgetpu.tflite")
print(info)  # {'status': 'loaded', 'input_shape': [1,224,224,3], ...}

# Run inference
input_data = np.zeros((1, 224, 224, 3), dtype=np.uint8)
output = client.infer(input_data)

# Or get embeddings from feature extraction models
embedding = client.get_embedding(input_data)

client.close()
```

The client also supports context managers:

```python
with EdgeTPUClient() as client:
    client.load_model("/path/to/model_edgetpu.tflite")
    output = client.infer(input_data)
```

### Handling busy errors

When multiple clients are active and the inference queue is full, methods raise `EdgeTPUBusyError`:

```python
import time
from rpi_edgetpu import EdgeTPUClient, EdgeTPUBusyError

with EdgeTPUClient() as client:
    client.load_model("/path/to/model_edgetpu.tflite")
    for attempt in range(3):
        try:
            output = client.infer(input_data)
            break
        except EdgeTPUBusyError:
            time.sleep(0.1)
```

## CLI Commands

The `edgetpu-cli` tool lets you interact with the service from the command line. Useful for shell scripts and LLM tool integrations.

```bash
# Check if service is alive
edgetpu-cli ping

# Show systemd service status
edgetpu-cli status

# Load a model onto the TPU
edgetpu-cli load-model /path/to/model_edgetpu.tflite

# Run inference (input/output as .npy files)
edgetpu-cli infer input.npy -o output.npy

# Extract embedding
edgetpu-cli embedding input.npy -o embedding.npy --shape 1,1280

# Re-scan for newly connected TPU devices (hot-plug)
edgetpu-cli rescan-tpus

# Machine-readable JSON output (all commands)
edgetpu-cli --json ping
edgetpu-cli --json infer input.npy
```

Without `-o`, `infer` and `embedding` print a shape/dtype/values summary to stdout. With `--json`, all commands produce JSON output.

## Example with picamera2

```python
import time
import numpy as np
from picamera2 import Picamera2
from rpi_edgetpu import EdgeTPUClient

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (224, 224), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

with EdgeTPUClient() as client:
    client.load_model("/home/pi/models/efficientnet-edgetpu-S_quant_edgetpu.tflite")

    for i in range(10):
        frame = picam2.capture_array()
        input_data = np.expand_dims(frame, axis=0).astype(np.uint8)
        embedding = client.get_embedding(input_data)
        print(f"Frame {i}: embedding shape {embedding.shape}")

picam2.stop()
```

## Examples

See the [`examples/`](examples/) directory for complete, annotated scripts:

- [`basic_inference.py`](examples/basic_inference.py) — Load a model and run inference
- [`embedding_extraction.py`](examples/embedding_extraction.py) — Extract and normalize embeddings
- [`picamera2_inference.py`](examples/picamera2_inference.py) — Live camera inference with FPS tracking
- [`picamera2_detection.py`](examples/picamera2_detection.py) — Live camera object detection with bounding boxes
- [`picamera2_embedding.py`](examples/picamera2_embedding.py) — Live camera embedding extraction
- [`picamera2_pose.py`](examples/picamera2_pose.py) — Live camera pose estimation with MoveNet
- [`multi_client.py`](examples/multi_client.py) — Concurrent clients with multiple models and multi-TPU affinity routing

## Models

Pre-compiled Edge TPU models are available at [EdgeTPUModelZoo](https://github.com/ricardodeazambuja/EdgeTPUModelZoo). Only models with `edgetpu` in their filename (e.g. `*_edgetpu.tflite`) are compiled for the Google Coral Edge TPU USB Accelerator and will work with this service. Other `.tflite` models in that repository are standard TFLite models that have not been compiled for the Edge TPU.

## Manual Setup

For advanced users who prefer manual control, see [INSTALL_PYENV.md](INSTALL_PYENV.md) for pyenv installation and [EDGETPU_SERVICE.md](EDGETPU_SERVICE.md) for the full service/client protocol reference.

## Requirements

- Raspberry Pi with Raspberry Pi OS Trixie (64-bit / arm64)
- One or more Coral USB Accelerators (multiple TPUs are detected and used automatically)
- System Python 3.13 with system-installed `numpy` and `picamera2` (pre-installed on Trixie)

The service venv uses [feranick's builds](https://github.com/feranick/TFlite-builds) (TF 2.17.1-based), which fix segfaults with SSD detection models (`TFLite_Detection_PostProcess` custom op). These are installed automatically by `install.sh`:

- `tflite-runtime` 2.17.1 (wheel)
- `pycoral` 2.0.3 (wheel)
- `libedgetpu1-std` 16.0tf2.17.1 (deb)
- `numpy < 2` (required by `tflite-runtime`)

## Training Your Own Models

For a complete example of fine-tuning and deploying custom models on the Coral Edge TPU, see [Maple-Syrup-Pi-Camera](https://github.com/ricardodeazambuja/Maple-Syrup-Pi-Camera). That project targets an earlier Raspberry Pi (Pi Zero) but includes Jupyter notebooks that walk through the full training-to-Edge-TPU pipeline.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
