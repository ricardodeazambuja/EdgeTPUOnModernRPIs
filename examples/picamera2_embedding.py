#!/usr/bin/env python3
"""
Live camera embedding extraction with picamera2.

Captures frames from a Raspberry Pi camera and extracts embeddings via the
Edge TPU. Prints the embedding shape, first few values, and FPS.

Requires picamera2 (pre-installed on Raspberry Pi OS).

Usage:
    python picamera2_embedding.py
"""

import time
import numpy as np
from picamera2 import Picamera2
from rpi_edgetpu import EdgeTPUClient

MODEL_PATH = "/home/pi/models/efficientnet-edgetpu-S_quant_embedding_extractor_edgetpu.tflite"
INPUT_SIZE = (224, 224)
NUM_FRAMES = 100

# Shape of the embedding layer to extract.
# Common values: [1, 1280] for EfficientNet, [1, 1024] for MobileNetV3-Large
EMBEDDING_SHAPE = [1, 1280]

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

        # Extract embedding
        embedding = client.get_embedding(input_data, embedding_shape=EMBEDDING_SHAPE)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm

        if i % 10 == 0:
            elapsed = time.monotonic() - start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"Frame {i}: embedding shape={embedding.shape} "
                f"first 5={embedding.flatten()[:5]} ({fps:.1f} FPS)"
            )

    elapsed = time.monotonic() - start
    print(f"\n{NUM_FRAMES} frames in {elapsed:.2f}s ({NUM_FRAMES / elapsed:.1f} FPS)")

picam2.stop()
