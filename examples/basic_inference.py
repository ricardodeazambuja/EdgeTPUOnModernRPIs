#!/usr/bin/env python3
"""
Basic Edge TPU inference example.

Loads a model, runs inference with dummy data, and prints the output.
Replace the model path and input shape/dtype with your own model's requirements.

Usage:
    python basic_inference.py
"""

import numpy as np
from rpi_edgetpu import EdgeTPUClient

# Path to your compiled Edge TPU model
MODEL_PATH = "/home/pi/models/mobilenet_v2_1.0_224_quant_edgetpu.tflite"

with EdgeTPUClient() as client:
    # Load model — only needs to happen once; subsequent calls are no-ops
    info = client.load_model(MODEL_PATH)
    print(f"Model loaded: {info}")
    # Example output:
    #   {'status': 'loaded', 'input_shape': [1, 224, 224, 3],
    #    'input_dtype': '<class numpy.uint8>', 'output_shape': [1, 1001]}

    # Create dummy input matching the model's expected shape and dtype
    input_data = np.zeros(info["input_shape"], dtype=np.uint8)

    # Run inference
    output = client.infer(input_data)
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Top class: {np.argmax(output)}, score: {output.max()}")
