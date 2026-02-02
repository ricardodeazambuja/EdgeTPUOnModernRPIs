#!/usr/bin/env bash
set -euo pipefail

# ─── Edge TPU Service Installer ───────────────────────────────────────────────
# Self-contained setup script for the Coral Edge TPU inference service.
#
# Usage:
#   bash install.sh              # interactive install
#   bash install.sh install      # same as above
#   bash install.sh uninstall    # teardown
#   bash install.sh status       # check service
#   bash install.sh update       # re-deploy service script & restart
#
# Or via curl:
#   curl -fsSL https://raw.githubusercontent.com/<user>/<repo>/main/install.sh | bash
# ──────────────────────────────────────────────────────────────────────────────

PYENV_PYTHON_VERSION="${PYENV_PYTHON_VERSION:-3.11.9}"
CORAL_VENV="${CORAL_VENV:-$HOME/.coral-venv}"
TFLITE_WHL_URL="https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs/releases/download/2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_aarch64.whl"
LIBEDGETPU_DEB_URL="https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs/releases/download/2.17.1/libedgetpu1-std_16.0tf2.17.1-1.trixie_arm64.deb"
PYCORAL_WHL_URL="https://github.com/ricardodeazambuja/EdgeTPUOnModernRPIs/releases/download/2.17.1/pycoral-2.0.3-cp311-cp311-linux_aarch64.whl"
CORAL_SERVICE_DIR="${CORAL_SERVICE_DIR:-$HOME/coral-service}"
SERVICE_NAME="edgetpu"
SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
SOCKET_PATH="/tmp/edgetpu.sock"
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"

APT_PACKAGES=(
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev
    libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m>>>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
skip()  { printf '\033[1;36m⏭\033[0m %s (already done)\n' "$*"; }

confirm() {
    local prompt="$1"
    # If stdin is not a terminal (piped), default to yes
    if [[ ! -t 0 ]]; then
        return 0
    fi
    local reply
    read -r -p "$prompt [Y/n] " reply
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ─── Embedded service.py ─────────────────────────────────────────────────────
# This function writes the service script so no repo clone is needed.

write_service_py() {
    local dest="$1"
    local venv_python="$2"
    cat > "$dest" << 'SERVICEEOF'
#!/usr/bin/env python3
"""
Edge TPU inference service with multi-TPU support.
Runs as systemd service on Python 3.11 with tflite-runtime.
Communicates via Unix socket with binary numpy arrays.
Supports multiple concurrent clients via per-TPU queues + dedicated inference worker threads.
Routes models to TPUs with affinity (same model reuses its TPU) and LRU eviction.
"""

import os
import socket
import struct
import json
import threading
import queue
import time
import concurrent.futures
import numpy as np
import tflite_runtime.interpreter as tflite

SOCKET_PATH = "/tmp/edgetpu.sock"
MAX_QUEUE_SIZE = 16

# Exception strings that indicate a TPU hardware error (device disconnected, etc.)
_TPU_DEVICE_ERROR_SUBSTRINGS = (
    "Failed to load delegate",
    "Failed to invoke",
    "Could not open",
    "USB transfer error",
    "device not found",
    "edgetpu",
    "libedgetpu",
    "delegate",
    "does not satisfy device ordinal",
)


class EdgeTPUService:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = None

    def load_model(self, model_path):
        """Load model onto Edge TPU. Call once, reuse forever."""
        if self.model_path == model_path and self.interpreter:
            return {"status": "already_loaded"}

        try:
            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[
                    tflite.load_delegate(
                        'libedgetpu.so.1',
                        options={"device": f":{self.device_index}"}
                    )
                ]
            )
            interpreter.allocate_tensors()
        except Exception as e:
            # Keep previous interpreter/model intact on failure
            return {"error": "load_failed", "message": str(e)}

        # Success — commit the new interpreter
        self.interpreter = interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_path = model_path

        return {
            "status": "loaded",
            "input_shape": self.input_details[0]['shape'].tolist(),
            "input_dtype": str(self.input_details[0]['dtype']),
            "output_shape": self.output_details[0]['shape'].tolist()
        }

    def infer(self, input_data):
        """Run inference on pre-loaded model."""
        if not self.interpreter:
            return {"error": "no model loaded"}

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output

    def detect(self, input_data):
        """Run SSD detection model and return all output tensors.

        SSD models with TFLite_Detection_PostProcess produce 4 outputs:
          0: boxes     [1, N, 4]  — bounding boxes (ymin, xmin, ymax, xmax) normalized 0-1
          1: classes   [1, N]     — class indices (float)
          2: scores    [1, N]     — confidence scores
          3: count     [1]        — number of valid detections
        """
        if not self.interpreter:
            return {"error": "no model loaded"}

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        outputs = {}
        for i, detail in enumerate(self.output_details):
            tensor = self.interpreter.get_tensor(detail['index'])
            outputs[f"output_{i}"] = tensor.tolist()
            outputs[f"output_{i}_shape"] = list(tensor.shape)

        outputs["num_outputs"] = len(self.output_details)
        return outputs

    def get_embedding(self, input_data, embedding_shape=[1, 1280]):
        """Get intermediate embedding layer (for feature extraction models)."""
        if not self.interpreter:
            return {"error": "no model loaded"}

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Find embedding layer by shape
        all_layers = self.interpreter.get_tensor_details()
        for t in all_layers:
            if t['shape'].tolist() == embedding_shape:
                return self.interpreter.get_tensor(t['index'])

        return {"error": f"no layer with shape {embedding_shape}"}


class TPUSlot:
    """Represents one physical Edge TPU with its own service, queue, and worker."""

    def __init__(self, device_index):
        self.device_index = device_index
        self.service = EdgeTPUService(device_index)
        self.queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.model_path = None
        self.last_used = 0.0
        self.alive = True


class TPUManager:
    """Routes inference requests to TPU slots with model affinity and LRU eviction."""

    def __init__(self):
        self.slots = []
        self.lock = threading.Lock()
        self._workers = []
        self._rescan_in_progress = False
        self.rescan()

    def rescan(self):
        """Probe for TPU devices. Additive: keep existing slots, add new ones."""
        indices = probe_tpu_devices()
        with self.lock:
            existing = {s.device_index for s in self.slots if s.alive}
            new_indices = [i for i in indices if i not in existing]

            # Mark slots for devices no longer present as dead
            for slot in self.slots:
                if slot.device_index not in indices:
                    slot.alive = False

            # Add new slots and start workers
            for idx in new_indices:
                slot = TPUSlot(idx)
                self.slots.append(slot)
                worker = threading.Thread(
                    target=inference_worker,
                    args=(slot, self),
                    daemon=True,
                    name=f"tpu-worker-{idx}",
                )
                worker.start()
                self._workers.append(worker)
                print(f"Started worker for TPU :{idx}")

            alive_count = sum(1 for s in self.slots if s.alive)
            print(f"TPU rescan complete: {alive_count} device(s) available")
            return alive_count

    def rescan_async(self):
        """Trigger a rescan on a background thread (deduplicated)."""
        with self.lock:
            if self._rescan_in_progress:
                return
            self._rescan_in_progress = True

        def _do_rescan():
            try:
                self.rescan()
            finally:
                with self.lock:
                    self._rescan_in_progress = False

        threading.Thread(target=_do_rescan, daemon=True, name="tpu-rescan").start()

    def mark_slot_dead(self, slot):
        """Mark a slot as dead and drain its queue."""
        with self.lock:
            slot.alive = False
            slot.model_path = None
        # Drain pending requests so waiting clients get immediate errors
        while True:
            try:
                request, future = slot.queue.get_nowait()
                if not future.done():
                    future.set_exception(
                        RuntimeError("TPU device disconnected")
                    )
                slot.queue.task_done()
            except queue.Empty:
                break

    def route(self, model_path):
        """Return slot index for a model. Priority: affinity → empty → LRU eviction."""
        with self.lock:
            alive_slots = [(i, s) for i, s in enumerate(self.slots) if s.alive]
            if not alive_slots:
                raise RuntimeError("No TPU devices available")

            # 1. Affinity: find slot already loaded with this model
            for i, s in alive_slots:
                if s.model_path == model_path:
                    return i

            # 2. Empty slot: find one with no model loaded
            for i, s in alive_slots:
                if s.model_path is None:
                    return i

            # 3. LRU eviction: pick the slot used least recently
            lru_idx = min(alive_slots, key=lambda x: x[1].last_used)[0]
            return lru_idx

    def update_slot_model(self, idx, model_path):
        """Called by worker after load_model succeeds."""
        with self.lock:
            self.slots[idx].model_path = model_path
            self.slots[idx].last_used = time.monotonic()

    def touch_slot(self, idx):
        """Update last_used timestamp on inference."""
        with self.lock:
            self.slots[idx].last_used = time.monotonic()

    def get_slot_model(self, idx):
        """Return the model currently loaded on a slot."""
        with self.lock:
            return self.slots[idx].model_path

    def slot_count(self):
        """Return number of alive slots."""
        with self.lock:
            return sum(1 for s in self.slots if s.alive)


def probe_tpu_devices():
    """Return list of valid device indices by trying load_delegate."""
    indices = []
    for i in range(8):
        try:
            delegate = tflite.load_delegate(
                'libedgetpu.so.1', options={"device": f":{i}"}
            )
            indices.append(i)
            del delegate
        except (ValueError, RuntimeError):
            break
    return indices if indices else [0]


def recv_exact(sock, n):
    """Receive exactly n bytes."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def _looks_like_device_error(exc):
    """Return True if the exception looks like a TPU hardware/USB error."""
    if not isinstance(exc, (RuntimeError, ValueError, OSError)):
        return False
    msg = str(exc).lower()
    return any(s.lower() in msg for s in _TPU_DEVICE_ERROR_SUBSTRINGS)


def send_error_response(conn, command, error_dict):
    """Send an error response using the correct wire format for *command*.

    For infer/embedding: prefix with 4-byte zero (signals JSON-not-numpy), then JSON.
    For everything else: just length-prefixed JSON.
    """
    resp_bytes = json.dumps(error_dict).encode('utf-8')
    if command in ('infer', 'embedding'):
        conn.sendall(struct.pack('!I', 0))
        conn.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)
    else:
        conn.sendall(struct.pack('!I', len(resp_bytes)) + resp_bytes)


def inference_worker(slot, manager):
    """Runs in dedicated thread per TPU. Processes inference requests serially."""
    while True:
        request, future = slot.queue.get()
        if not slot.alive:
            future.set_exception(RuntimeError("TPU device removed"))
            slot.queue.task_done()
            continue
        try:
            command = request['command']
            if command == 'load_model':
                result = slot.service.load_model(request['model_path'])
                if isinstance(result, dict) and result.get('status') in ('loaded', 'already_loaded'):
                    manager.update_slot_model(
                        request['_slot_idx'], request['model_path']
                    )
            elif command == 'infer':
                manager.touch_slot(request['_slot_idx'])
                result = slot.service.infer(request['input_data'])
            elif command == 'detect':
                manager.touch_slot(request['_slot_idx'])
                result = slot.service.detect(request['input_data'])
            elif command == 'embedding':
                manager.touch_slot(request['_slot_idx'])
                result = slot.service.get_embedding(
                    request['input_data'],
                    request.get('embedding_shape', [1, 1280])
                )
            elif command == 'ping':
                result = {'status': 'ok'}
            else:
                result = {'error': f'unknown command: {command}'}
            future.set_result(result)
        except Exception as e:
            if _looks_like_device_error(e):
                print(f"TPU :{slot.device_index} device error: {e}")
                manager.mark_slot_dead(slot)
                manager.rescan_async()
                future.set_exception(
                    RuntimeError("tpu_disconnected")
                )
            else:
                future.set_exception(e)
        finally:
            slot.queue.task_done()


def handle_client(conn, manager):
    """Handle one client connection (persistent). Routes requests to TPU slots."""
    client_model_path = None
    client_tpu_idx = None

    try:
        while True:
            # Protocol: [4-byte header length][JSON header][optional numpy bytes]
            header_len_bytes = recv_exact(conn, 4)
            header_len = struct.unpack('!I', header_len_bytes)[0]

            header_json = recv_exact(conn, header_len).decode('utf-8')
            header = json.loads(header_json)

            command = header.get('command')

            if command == 'quit':
                break

            try:
                # Build request dict, deserializing numpy data on the client thread
                request = {'command': command}

                if command == 'load_model':
                    model_path = header['model_path']
                    request['model_path'] = model_path
                    # Route to appropriate TPU
                    tpu_idx = manager.route(model_path)
                    request['_slot_idx'] = tpu_idx

                elif command in ('infer', 'detect', 'embedding'):
                    if client_model_path is None:
                        error_resp = {
                            "error": "no_model",
                            "message": "Call load_model before inference"
                        }
                        # Consume the numpy data from socket first
                        array_bytes = recv_exact(conn, header['data_size'])
                        send_error_response(conn, command, error_resp)
                        continue

                    # Check if our model was evicted from the slot
                    current_model = manager.get_slot_model(client_tpu_idx)
                    if current_model != client_model_path:
                        # Re-route: find a slot for our model
                        client_tpu_idx = manager.route(client_model_path)
                        # Inject a load_model before the actual command
                        load_req = {
                            'command': 'load_model',
                            'model_path': client_model_path,
                            '_slot_idx': client_tpu_idx,
                        }
                        load_future = concurrent.futures.Future()
                        try:
                            manager.slots[client_tpu_idx].queue.put(
                                (load_req, load_future), block=False
                            )
                        except queue.Full:
                            error_resp = {
                                "error": "server_busy",
                                "message": "Inference queue is full, try again later"
                            }
                            # Consume numpy data
                            array_bytes = recv_exact(conn, header['data_size'])
                            send_error_response(conn, command, error_resp)
                            continue
                        # Wait for reload
                        try:
                            load_future.result()
                        except Exception as e:
                            error_resp = {"error": str(e)}
                            array_bytes = recv_exact(conn, header['data_size'])
                            send_error_response(conn, command, error_resp)
                            continue

                    array_bytes = recv_exact(conn, header['data_size'])
                    input_data = np.frombuffer(array_bytes, dtype=header['dtype'])
                    input_data = input_data.reshape(header['shape'])
                    request['input_data'] = input_data
                    request['_slot_idx'] = client_tpu_idx
                    if command == 'embedding':
                        request['embedding_shape'] = header.get('embedding_shape', [1, 1280])

                elif command == 'rescan_tpus':
                    count = manager.rescan()
                    response = json.dumps({
                        "status": "ok",
                        "tpu_count": count,
                    }).encode('utf-8')
                    conn.sendall(struct.pack('!I', len(response)) + response)
                    continue

                elif command == 'ping':
                    request['_slot_idx'] = 0

                # Determine which slot queue to use
                if command == 'load_model':
                    target_idx = tpu_idx
                elif command in ('infer', 'detect', 'embedding'):
                    target_idx = client_tpu_idx
                elif command == 'ping':
                    target_idx = 0
                else:
                    target_idx = 0

                # Enqueue with backpressure
                future = concurrent.futures.Future()
                try:
                    manager.slots[target_idx].queue.put((request, future), block=False)
                except queue.Full:
                    error_resp = {
                        "error": "server_busy",
                        "message": "Inference queue is full, try again later"
                    }
                    send_error_response(conn, command, error_resp)
                    continue

                # Wait for worker to process
                try:
                    result = future.result()
                except Exception as e:
                    error_resp = {"error": str(e)}
                    send_error_response(conn, command, error_resp)
                    continue

                # For load_model: only commit client state after success
                if command == 'load_model':
                    if isinstance(result, dict) and result.get('status') in ('loaded', 'already_loaded'):
                        client_model_path = model_path
                        client_tpu_idx = tpu_idx

                # Send response back using the same wire format as before
                if command in ('load_model', 'ping', 'detect'):
                    response = json.dumps(result).encode('utf-8')
                    conn.sendall(struct.pack('!I', len(response)) + response)

                elif command in ('infer', 'embedding'):
                    if isinstance(result, dict):  # Error
                        response = json.dumps(result).encode('utf-8')
                        conn.sendall(struct.pack('!I', 0))  # 0 = JSON response
                        conn.sendall(struct.pack('!I', len(response)) + response)
                    else:
                        out_bytes = result.tobytes()
                        out_header = json.dumps({
                            'shape': result.shape,
                            'dtype': str(result.dtype)
                        }).encode('utf-8')
                        conn.sendall(struct.pack('!I', len(out_bytes)))  # >0 = numpy
                        conn.sendall(struct.pack('!I', len(out_header)) + out_header)
                        conn.sendall(out_bytes)

                else:
                    # Unknown command — worker already returned an error dict
                    response = json.dumps(result).encode('utf-8')
                    conn.sendall(struct.pack('!I', len(response)) + response)

            except ConnectionError:
                raise  # Re-raise so outer handler closes connection
            except Exception as e:
                # Unexpected error processing this request — send error, keep connection
                try:
                    send_error_response(conn, command, {
                        "error": "internal_error",
                        "message": str(e)
                    })
                except ConnectionError:
                    raise
                except Exception:
                    pass  # Can't send error either, but don't kill the loop

    except ConnectionError:
        pass  # Client disconnected
    finally:
        conn.close()


def main():
    # Clean up old socket
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    manager = TPUManager()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)  # Allow all users
    server.listen(5)

    print(f"Edge TPU service listening on {SOCKET_PATH}")

    while True:
        conn, _ = server.accept()
        print("Client connected")
        client_thread = threading.Thread(
            target=handle_client,
            args=(conn, manager),
            daemon=True
        )
        client_thread.start()


if __name__ == "__main__":
    main()
SERVICEEOF
    # Replace the shebang with the venv python path
    sed -i "1s|#!/usr/bin/env python3|#!${venv_python}|" "$dest"
    chmod 755 "$dest"
}

# ─── Step functions ───────────────────────────────────────────────────────────

step_apt_deps() {
    info "Step 1/6: Install build dependencies and libedgetpu"
    echo
    echo "  The following packages are needed to compile Python from source:"
    echo "    ${APT_PACKAGES[*]}"
    echo
    echo "  libedgetpu1-std (TF 2.17.1, originally from feranick's build) is installed"
    echo "  from our GitHub release."
    echo
    echo "  This requires sudo."
    echo

    # Check if apt packages are already installed
    local apt_installed=true
    for pkg in "${APT_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            apt_installed=false
            break
        fi
    done

    # Check if libedgetpu is installed at the right version
    local edgetpu_installed=false
    if dpkg -s libedgetpu1-std &>/dev/null; then
        local installed_ver
        installed_ver="$(dpkg-query -W -f='${Version}' libedgetpu1-std 2>/dev/null || true)"
        if [[ "$installed_ver" == *"16.0tf2.17.1"* ]]; then
            edgetpu_installed=true
        fi
    fi

    if $apt_installed && $edgetpu_installed; then
        skip "Build dependencies and libedgetpu"
        return
    fi

    if ! confirm "  Install build dependencies and libedgetpu?"; then
        warn "Skipped dependencies. Later steps may fail."
        return
    fi

    if ! $apt_installed; then
        sudo apt update
        sudo apt install -y "${APT_PACKAGES[@]}"
    fi

    if ! $edgetpu_installed; then
        info "Installing libedgetpu (TF 2.17.1)..."
        local tmp_deb="/tmp/libedgetpu1-std.deb"
        curl -fsSL -o "$tmp_deb" "$LIBEDGETPU_DEB_URL"
        sudo dpkg -i "$tmp_deb"
        rm -f "$tmp_deb"
    fi

    ok "Build dependencies and libedgetpu installed"
}

step_pyenv() {
    info "Step 2/6: Install pyenv"
    echo
    echo "  pyenv manages Python versions without touching the system Python."
    echo "  It will be installed to ${PYENV_ROOT} and init lines added to ~/.bashrc."
    echo

    local pyenv_bin="${PYENV_ROOT}/bin/pyenv"

    if [[ -x "$pyenv_bin" ]]; then
        skip "pyenv (found at ${pyenv_bin})"
        return
    fi

    if ! confirm "  Install pyenv?"; then
        warn "Skipped pyenv. Cannot continue without it."
        return
    fi

    curl https://pyenv.run | bash

    # Add to bashrc if not already there
    local bashrc="$HOME/.bashrc"
    local marker='export PYENV_ROOT="$HOME/.pyenv"'
    if [[ -f "$bashrc" ]] && ! grep -qF "$marker" "$bashrc"; then
        {
            echo ''
            echo '# pyenv (added by edgetpu installer)'
            echo 'export PYENV_ROOT="$HOME/.pyenv"'
            echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
            echo 'eval "$(pyenv init -)"'
        } >> "$bashrc"
    fi

    if [[ ! -x "$pyenv_bin" ]]; then
        err "pyenv installation failed — ${pyenv_bin} not found"
        exit 1
    fi

    ok "pyenv installed"
}

step_python() {
    info "Step 3/6: Build Python ${PYENV_PYTHON_VERSION} via pyenv"
    echo
    echo "  WARNING: This compiles Python from source and will take a while"
    echo "  (10-20 minutes on a Raspberry Pi)."
    echo

    local pyenv_bin="${PYENV_ROOT}/bin/pyenv"
    local python_bin="${PYENV_ROOT}/versions/${PYENV_PYTHON_VERSION}/bin/python"

    if [[ -x "$python_bin" ]]; then
        skip "Python ${PYENV_PYTHON_VERSION} (found at ${python_bin})"
        return
    fi

    if [[ ! -x "$pyenv_bin" ]]; then
        err "pyenv not found at ${pyenv_bin}. Run step 2 first."
        exit 1
    fi

    if ! confirm "  Build Python ${PYENV_PYTHON_VERSION}?"; then
        warn "Skipped Python build. Cannot continue without it."
        return
    fi

    export PYENV_ROOT
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    "$pyenv_bin" install "$PYENV_PYTHON_VERSION"

    if [[ ! -x "$python_bin" ]]; then
        err "Python build failed — ${python_bin} not found"
        exit 1
    fi

    ok "Python ${PYENV_PYTHON_VERSION} installed"
}

step_venv() {
    info "Step 4/6: Create venv and install dependencies"
    echo
    echo "  Creates ${CORAL_VENV} with Python ${PYENV_PYTHON_VERSION}."
    echo "  Installs tflite-runtime 2.17.1, pycoral 2.0.3, and numpy<2"
    echo "  from our GitHub release (originally feranick's builds)."
    echo

    local python_bin="${PYENV_ROOT}/versions/${PYENV_PYTHON_VERSION}/bin/python"

    if [[ -x "${CORAL_VENV}/bin/python" ]]; then
        skip "venv (found at ${CORAL_VENV})"
        return
    fi

    if [[ ! -x "$python_bin" ]]; then
        err "Python ${PYENV_PYTHON_VERSION} not found. Run step 3 first."
        exit 1
    fi

    if ! confirm "  Create venv and install dependencies?"; then
        warn "Skipped venv creation."
        return
    fi

    "$python_bin" -m venv "$CORAL_VENV"
    "${CORAL_VENV}/bin/pip" install --upgrade pip
    "${CORAL_VENV}/bin/pip" install "numpy<2" "$TFLITE_WHL_URL" "$PYCORAL_WHL_URL"

    ok "venv created and dependencies installed"
}

step_deploy() {
    info "Step 5/6: Deploy service script"
    echo
    echo "  Writes service.py (embedded in this script) to ${CORAL_SERVICE_DIR}/."
    echo

    local dest="${CORAL_SERVICE_DIR}/edgetpu_service.py"
    local venv_python="${CORAL_VENV}/bin/python"

    if [[ -f "$dest" ]]; then
        skip "Service script (found at ${dest})"
        return
    fi

    if ! confirm "  Deploy service script?"; then
        warn "Skipped service script deployment."
        return
    fi

    mkdir -p "$CORAL_SERVICE_DIR"
    write_service_py "$dest" "$venv_python"

    ok "Service script deployed to ${dest}"
}

step_systemd() {
    info "Step 6/6: Install and start systemd service"
    echo
    echo "  Requires sudo for systemctl."
    echo "  Installs ${SYSTEMD_UNIT}, enables and starts the service."
    echo

    local dest="${CORAL_SERVICE_DIR}/edgetpu_service.py"
    local venv_python="${CORAL_VENV}/bin/python"
    local user
    user="$(whoami)"

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        skip "systemd service (already active)"
        return
    fi

    if [[ ! -f "$dest" ]]; then
        err "Service script not found at ${dest}. Run step 5 first."
        exit 1
    fi

    if ! confirm "  Install and start systemd service?"; then
        warn "Skipped systemd service installation."
        return
    fi

    local unit_content
    unit_content="[Unit]
Description=Edge TPU Inference Service
After=network.target

[Service]
Type=simple
User=${user}
ExecStart=${venv_python} ${dest}
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target"

    local tmp="/tmp/edgetpu.service"
    echo "$unit_content" > "$tmp"
    sudo cp "$tmp" "$SYSTEMD_UNIT"
    rm -f "$tmp"

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    sudo systemctl start "$SERVICE_NAME"

    ok "systemd service '${SERVICE_NAME}' installed and started"
}

# ─── Commands ─────────────────────────────────────────────────────────────────

do_install() {
    echo
    echo "══════════════════════════════════════════════════"
    echo "  Edge TPU Service Installer"
    echo "══════════════════════════════════════════════════"
    echo

    step_apt_deps
    echo
    step_pyenv
    echo
    step_python
    echo
    step_venv
    echo
    step_deploy
    echo
    step_systemd

    echo
    echo "══════════════════════════════════════════════════"
    echo "  Setup complete!"
    echo "══════════════════════════════════════════════════"
    echo
    echo "  Service: sudo systemctl status ${SERVICE_NAME}"
    echo "  Logs:    journalctl -u ${SERVICE_NAME} -f"
    echo "  Client:  pip install git+https://github.com/<user>/<repo>.git"
    echo "           from rpi_edgetpu import EdgeTPUClient"
    echo
}

do_uninstall() {
    echo
    echo "══════════════════════════════════════════════════"
    echo "  Edge TPU Service Uninstaller"
    echo "══════════════════════════════════════════════════"
    echo

    # Stop and disable service
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        info "Stopping service..."
        sudo systemctl stop "$SERVICE_NAME"
        ok "Service stopped"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        sudo systemctl disable "$SERVICE_NAME"
        ok "Service disabled"
    fi

    # Remove systemd unit
    if [[ -f "$SYSTEMD_UNIT" ]]; then
        sudo rm -f "$SYSTEMD_UNIT"
        sudo systemctl daemon-reload
        ok "Systemd unit removed"
    fi

    # Remove service directory
    if [[ -d "$CORAL_SERVICE_DIR" ]]; then
        rm -rf "$CORAL_SERVICE_DIR"
        ok "Removed ${CORAL_SERVICE_DIR}"
    fi

    # Remove venv
    if [[ -d "$CORAL_VENV" ]]; then
        rm -rf "$CORAL_VENV"
        ok "Removed ${CORAL_VENV}"
    fi

    # Remove socket
    if [[ -S "$SOCKET_PATH" ]]; then
        rm -f "$SOCKET_PATH"
        ok "Removed socket"
    fi

    echo
    echo "Uninstall complete."
    echo "Note: pyenv was NOT removed. Remove it manually if desired:"
    echo "  rm -rf ~/.pyenv"
    echo
}

do_status() {
    echo
    info "Service status:"
    systemctl status "$SERVICE_NAME" --no-pager 2>/dev/null || true
    echo
    info "Socket:"
    if [[ -S "$SOCKET_PATH" ]]; then
        echo "  ${SOCKET_PATH} exists"
    else
        echo "  ${SOCKET_PATH} not found"
    fi
    echo
    info "Venv:"
    if [[ -d "$CORAL_VENV" ]]; then
        echo "  ${CORAL_VENV} exists"
    else
        echo "  ${CORAL_VENV} not found"
    fi
    echo
    info "Service script:"
    if [[ -f "${CORAL_SERVICE_DIR}/edgetpu_service.py" ]]; then
        echo "  ${CORAL_SERVICE_DIR}/edgetpu_service.py exists"
    else
        echo "  ${CORAL_SERVICE_DIR}/edgetpu_service.py not found"
    fi
    echo
}

do_update() {
    echo
    echo "══════════════════════════════════════════════════"
    echo "  Edge TPU Service Update"
    echo "══════════════════════════════════════════════════"
    echo

    local dest="${CORAL_SERVICE_DIR}/edgetpu_service.py"
    local venv_python="${CORAL_VENV}/bin/python"

    if [[ ! -d "$CORAL_SERVICE_DIR" ]]; then
        err "Service directory ${CORAL_SERVICE_DIR} not found. Run 'bash install.sh install' first."
        exit 1
    fi

    if [[ ! -x "$venv_python" ]]; then
        err "Venv python not found at ${venv_python}. Run 'bash install.sh install' first."
        exit 1
    fi

    # Upgrade libedgetpu deb
    info "Upgrading libedgetpu (TF 2.17.1)..."
    local tmp_deb="/tmp/libedgetpu1-std.deb"
    curl -fsSL -o "$tmp_deb" "$LIBEDGETPU_DEB_URL"
    sudo dpkg -i "$tmp_deb"
    rm -f "$tmp_deb"
    ok "libedgetpu upgraded"

    # Upgrade venv packages
    info "Upgrading venv packages (tflite-runtime, pycoral)..."
    "${CORAL_VENV}/bin/pip" install --upgrade "numpy<2" "$TFLITE_WHL_URL" "$PYCORAL_WHL_URL"
    ok "Venv packages upgraded"

    info "Re-deploying service script to ${dest}..."
    write_service_py "$dest" "$venv_python"
    ok "Service script updated"

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        info "Restarting ${SERVICE_NAME} service..."
        sudo systemctl restart "$SERVICE_NAME"
        ok "Service restarted"
    else
        warn "Service '${SERVICE_NAME}' is not running. Start it with: sudo systemctl start ${SERVICE_NAME}"
    fi

    echo
}

# ─── Main ─────────────────────────────────────────────────────────────────────

cmd="${1:-install}"

case "$cmd" in
    install)    do_install ;;
    uninstall)  do_uninstall ;;
    status)     do_status ;;
    update)     do_update ;;
    *)
        echo "Usage: $0 {install|uninstall|status|update}"
        exit 1
        ;;
esac
