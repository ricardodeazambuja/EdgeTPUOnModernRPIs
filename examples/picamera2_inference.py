#!/usr/bin/env python3
"""
Live camera inference with picamera2.

Captures frames from a Raspberry Pi camera and runs Edge TPU inference
on each frame. Prints the top classification result and FPS.

Requires picamera2 (pre-installed on Raspberry Pi OS).

Usage:
    python picamera2_inference.py
"""

import time
import numpy as np
from picamera2 import Picamera2
from rpi_edgetpu import EdgeTPUClient

MODEL_PATH = "/home/pi/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
INPUT_SIZE = (224, 224)
NUM_FRAMES = 100

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

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        input_data = np.expand_dims(frame, axis=0).astype(np.uint8)

        output = client.infer(input_data)
        top_class = np.argmax(output)
        score = output.max()

        if i % 10 == 0:
            elapsed = time.monotonic() - start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Frame {i}: class={top_class} score={score:.3f} ({fps:.1f} FPS)")

    elapsed = time.monotonic() - start
    print(f"\n{NUM_FRAMES} frames in {elapsed:.2f}s ({NUM_FRAMES / elapsed:.1f} FPS)")

picam2.stop()
