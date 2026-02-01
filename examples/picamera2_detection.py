#!/usr/bin/env python3
"""
Live camera object detection with picamera2.

Captures frames from a Raspberry Pi camera and runs SSD object detection
on the Edge TPU. Prints detected objects with bounding boxes and FPS.

Requires picamera2 (pre-installed on Raspberry Pi OS) and an SSD model
with TFLite_Detection_PostProcess built in (e.g. ssd_mobilenet_v2_coco).

Usage:
    python picamera2_detection.py
"""

import time
import numpy as np
from picamera2 import Picamera2
from rpi_edgetpu import EdgeTPUClient

MODEL_PATH = "/home/pi/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_PATH = "/home/pi/models/coco_labels.txt"
INPUT_SIZE = (300, 300)
NUM_FRAMES = 100
SCORE_THRESHOLD = 0.4
TOP_K = 10


def load_labels(path):
    """Load label file. Each line is either 'id label' or just 'label'."""
    labels = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]
                elif len(parts) == 1:
                    labels[len(labels)] = parts[0]
    except FileNotFoundError:
        pass
    return labels


labels = load_labels(LABELS_PATH)

# Set up camera to output frames at the model's expected resolution
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": INPUT_SIZE, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)  # Let the camera warm up

with EdgeTPUClient() as client:
    info = client.load_model(MODEL_PATH)
    print(f"Model loaded: {info}")

    start = time.monotonic()
    for i in range(NUM_FRAMES):
        # Capture a frame (already the right size thanks to camera config)
        frame = picam2.capture_array()

        # Add batch dimension: (300, 300, 3) -> (1, 300, 300, 3)
        input_data = np.expand_dims(frame, axis=0).astype(np.uint8)

        detections = client.detect(
            input_data,
            score_threshold=SCORE_THRESHOLD,
            top_k=TOP_K,
        )

        if i % 10 == 0:
            elapsed = time.monotonic() - start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\nFrame {i} ({fps:.1f} FPS) — {len(detections)} detection(s):")
            for det in detections:
                label = labels.get(det['class_id'], f"class_{det['class_id']}")
                b = det['bbox']
                print(
                    f"  {label}: {det['score']:.2f} "
                    f"[ymin={b['ymin']:.3f} xmin={b['xmin']:.3f} "
                    f"ymax={b['ymax']:.3f} xmax={b['xmax']:.3f}]"
                )

    elapsed = time.monotonic() - start
    print(f"\n{NUM_FRAMES} frames in {elapsed:.2f}s ({NUM_FRAMES / elapsed:.1f} FPS)")

picam2.stop()
