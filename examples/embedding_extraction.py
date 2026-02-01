#!/usr/bin/env python3
"""
Embedding extraction example.

Uses a feature extraction model (e.g. EfficientNet) to get intermediate
layer embeddings from the Edge TPU. Useful for similarity search, clustering,
or feeding into a downstream classifier.

Usage:
    python embedding_extraction.py
"""

import numpy as np
from rpi_edgetpu import EdgeTPUClient

MODEL_PATH = "/home/pi/models/efficientnet-edgetpu-S_quant_embedding_extractor_edgetpu.tflite"

# Shape of the embedding layer to extract.
# Common values: [1, 1280] for EfficientNet, [1, 1024] for MobileNetV3-Large
EMBEDDING_SHAPE = [1, 1280]

with EdgeTPUClient() as client:
    info = client.load_model(MODEL_PATH)
    print(f"Model loaded: {info}")

    # Dummy input — replace with real image data
    input_data = np.zeros(info["input_shape"], dtype=np.uint8)

    # Extract embedding
    embedding = client.get_embedding(input_data, embedding_shape=EMBEDDING_SHAPE)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 values): {embedding.flatten()[:5]}")

    # Normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        normalized = embedding / norm
        print(f"Normalized embedding (first 5): {normalized.flatten()[:5]}")
