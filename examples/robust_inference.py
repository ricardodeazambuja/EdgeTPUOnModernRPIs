#!/usr/bin/env python3
"""
Robust inference example with graceful error handling.

Demonstrates best practices for production use:
  - Catching EdgeTPUError for service-reported errors (bad model, TPU gone, etc.)
  - Catching EdgeTPUBusyError for queue-full backpressure (retry with backoff)
  - Detecting TPU disconnection and waiting for recovery via rescan
  - Reconnecting when the service itself restarts or the socket breaks
  - Distinguishing recoverable vs fatal errors so the application stays alive

Usage:
    python robust_inference.py
"""

import time
import numpy as np
from rpi_edgetpu import EdgeTPUClient, EdgeTPUError, EdgeTPUBusyError

MODEL_PATH = "/home/pi/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite"

MAX_RETRIES = 5
RETRY_BACKOFF = 0.1        # seconds, multiplied by attempt number
RECONNECT_DELAY = 2.0      # seconds between reconnection attempts
MAX_RECONNECT_ATTEMPTS = 5


def connect(socket_path=None):
    """Create a client connection, retrying if the service isn't up yet."""
    kwargs = {"socket_path": socket_path} if socket_path else {}
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        try:
            client = EdgeTPUClient(**kwargs)
            client.ping()
            return client
        except (ConnectionError, OSError) as e:
            print(f"Connection attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}: {e}")
            if attempt < MAX_RECONNECT_ATTEMPTS:
                time.sleep(RECONNECT_DELAY)
    raise ConnectionError(
        f"Could not connect to Edge TPU service after {MAX_RECONNECT_ATTEMPTS} attempts"
    )


def load_model(client, model_path):
    """Load a model, handling bad paths and TPU errors.

    Returns the model info dict on success, or None if the model can't be loaded.
    """
    try:
        info = client.load_model(model_path)
        print(f"Model loaded: {info['status']}")
        return info
    except EdgeTPUError as e:
        # Service told us the load failed (bad path, corrupt model, etc.)
        print(f"Failed to load model: {e}")
        return None


def infer_with_retry(client, input_data):
    """Run inference with retry on transient errors.

    Returns the output array on success.
    Raises EdgeTPUError for non-recoverable service errors.
    Raises ConnectionError if the service connection is lost.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.infer(input_data)

        except EdgeTPUBusyError:
            # Queue is full — back off and retry
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF * attempt
                print(f"  Queue full, retrying in {delay:.1f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                print(f"  Queue full after {MAX_RETRIES} attempts, giving up on this frame")
                return None

        except EdgeTPUError as e:
            # Service-reported error (e.g. TPU disconnected, no model loaded)
            # These won't resolve by retrying the same request
            raise

    return None


def main():
    client = connect()
    model_info = None

    try:
        # --- Load model (with example of handling a bad path first) ---

        # Try a bad path to show that the client stays usable afterward
        print("\n--- Attempting to load a non-existent model ---")
        bad_info = load_model(client, "/tmp/does_not_exist.tflite")
        assert bad_info is None, "Expected load to fail"
        print("Client is still connected and usable after a failed load.\n")

        # Now load the real model
        print("--- Loading the real model ---")
        model_info = load_model(client, MODEL_PATH)
        if model_info is None:
            print("Could not load model. Check the path and try again.")
            return

        # --- Run inference loop ---

        input_data = np.zeros(model_info["input_shape"], dtype=np.uint8)
        print(f"\n--- Running inference (input shape: {model_info['input_shape']}) ---")

        for i in range(20):
            try:
                output = infer_with_retry(client, input_data)
                if output is not None:
                    print(f"  Frame {i:3d}: top class={np.argmax(output):4d}  score={output.max():.4f}")
                else:
                    print(f"  Frame {i:3d}: skipped (queue busy)")

            except EdgeTPUError as e:
                # TPU disconnected or other service error
                error_msg = str(e)
                print(f"  Frame {i:3d}: service error — {error_msg}")

                if "tpu_disconnected" in error_msg:
                    # The service detected the TPU was yanked and is rescanning.
                    # Wait a bit, then try to rescan and reload.
                    print("  TPU disconnected. Waiting for rescan...")
                    time.sleep(3)
                    try:
                        result = client.rescan_tpus()
                        print(f"  Rescan result: {result}")
                        model_info = load_model(client, MODEL_PATH)
                        if model_info is None:
                            print("  Could not reload model after rescan. Stopping.")
                            break
                        print("  Model reloaded — resuming inference.")
                    except (EdgeTPUError, ConnectionError) as rescan_err:
                        print(f"  Rescan/reload failed: {rescan_err}. Stopping.")
                        break
                else:
                    # Some other service error — log and skip this frame
                    continue

            except ConnectionError:
                # Socket broke — service may have restarted
                print(f"  Frame {i:3d}: connection lost, reconnecting...")
                client = connect()
                model_info = load_model(client, MODEL_PATH)
                if model_info is None:
                    print("  Could not reload model after reconnect. Stopping.")
                    break

    finally:
        try:
            client.close()
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
