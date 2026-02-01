#!/usr/bin/env python3
"""
Multi-client concurrent inference example with multi-model support.

Demonstrates multiple threads using different models simultaneously.
With multiple TPUs, each model is routed to its own TPU via model affinity.
With a single TPU, the service handles automatic model reloading.

If the queue is full, EdgeTPUBusyError is raised and the client retries.

Usage:
    python multi_client.py
"""

import time
import threading
import numpy as np
from rpi_edgetpu import EdgeTPUClient, EdgeTPUBusyError

# Two different models to demonstrate multi-TPU affinity routing.
# With 2 TPUs: model A goes to TPU 0, model B goes to TPU 1 — no interference.
# With 1 TPU: the service detects eviction and auto-reloads the correct model.
MODEL_A = "/home/pi/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
MODEL_B = "/home/pi/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"

INFERENCES_PER_CLIENT = 10
MAX_RETRIES = 5


def classification_worker(client_id, model_path):
    """Run classification inferences using the given model."""
    with EdgeTPUClient() as client:
        info = client.load_model(model_path)
        print(f"[Client {client_id}] loaded {model_path.split('/')[-1]} -> {info}")

        input_shape = info["input_shape"]
        input_data = np.zeros(input_shape, dtype=np.uint8)

        for i in range(INFERENCES_PER_CLIENT):
            for attempt in range(MAX_RETRIES):
                try:
                    output = client.infer(input_data)
                    top_class = np.argmax(output)
                    print(f"[Client {client_id}] inference {i}: class={top_class}")
                    break
                except EdgeTPUBusyError:
                    print(f"[Client {client_id}] inference {i}: busy, retrying...")
                    time.sleep(0.05 * (attempt + 1))
            else:
                print(f"[Client {client_id}] inference {i}: failed after {MAX_RETRIES} retries")

    print(f"[Client {client_id}] done")


def detection_worker(client_id, model_path):
    """Run detection inferences using the given model."""
    with EdgeTPUClient() as client:
        info = client.load_model(model_path)
        print(f"[Client {client_id}] loaded {model_path.split('/')[-1]} -> {info}")

        input_shape = info["input_shape"]
        input_data = np.zeros(input_shape, dtype=np.uint8)

        for i in range(INFERENCES_PER_CLIENT):
            for attempt in range(MAX_RETRIES):
                try:
                    detections = client.detect(input_data, score_threshold=0.3)
                    print(f"[Client {client_id}] detect {i}: {len(detections)} objects")
                    break
                except EdgeTPUBusyError:
                    print(f"[Client {client_id}] inference {i}: busy, retrying...")
                    time.sleep(0.05 * (attempt + 1))
            else:
                print(f"[Client {client_id}] detect {i}: failed after {MAX_RETRIES} retries")

    print(f"[Client {client_id}] done")


# Launch client threads with different models
threads = []
start = time.monotonic()

# Two classification clients sharing model A
for cid in range(2):
    t = threading.Thread(target=classification_worker, args=(cid, MODEL_A))
    t.start()
    threads.append(t)

# Two detection clients sharing model B
for cid in range(2, 4):
    t = threading.Thread(target=detection_worker, args=(cid, MODEL_B))
    t.start()
    threads.append(t)

# Wait for all to finish
for t in threads:
    t.join()

elapsed = time.monotonic() - start
total = 4 * INFERENCES_PER_CLIENT
print(f"\n{total} total inferences across 4 clients (2 models) in {elapsed:.2f}s")
