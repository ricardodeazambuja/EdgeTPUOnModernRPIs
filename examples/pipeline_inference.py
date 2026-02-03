#!/usr/bin/env python3
"""
Pipeline inference — chain multiple Edge TPU models server-side.

The output of each model feeds directly into the next model on the server.
Only the initial input and final output travel over the Unix socket, so
intermediate tensors never leave the service process.

Replace MODEL_A and MODEL_B with your own *_edgetpu.tflite model paths.
The output shape of model A must match the input shape of model B.
"""

import numpy as np
from rpi_edgetpu import EdgeTPUClient

# Replace these with real model paths
MODEL_A = "/path/to/first_model_edgetpu.tflite"
MODEL_B = "/path/to/second_model_edgetpu.tflite"

# Create dummy input matching the first model's expected shape
input_data = np.zeros((1, 224, 224, 3), dtype=np.uint8)

with EdgeTPUClient() as client:
    output = client.pipeline([MODEL_A, MODEL_B], input_data)
    print(f"Pipeline output shape: {output.shape}")
    print(f"Pipeline output dtype: {output.dtype}")
    top_idx = output.flatten().argsort()[-5:][::-1]
    print(f"Top-5 indices: {top_idx.tolist()}")
