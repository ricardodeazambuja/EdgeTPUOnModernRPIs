#!/usr/bin/env python3
"""
Live camera pose estimation with picamera2.

Captures frames from a Raspberry Pi camera and runs MoveNet SinglePose
Lightning on the Edge TPU. Prints detected keypoints and skeleton
connections with FPS.

Requires picamera2 (pre-installed on Raspberry Pi OS).

Usage:
    python picamera2_pose.py
"""

import time
import numpy as np
from picamera2 import Picamera2
from rpi_edgetpu import EdgeTPUClient

MODEL_PATH = "/home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite"
INPUT_SIZE = (192, 192)
NUM_FRAMES = 100
CONFIDENCE_THRESHOLD = 0.3

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_CONNECTIONS = [
    (0, 1),   # nose – left_eye
    (0, 2),   # nose – right_eye
    (1, 3),   # left_eye – left_ear
    (2, 4),   # right_eye – right_ear
    (5, 6),   # left_shoulder – right_shoulder
    (5, 7),   # left_shoulder – left_elbow
    (6, 8),   # right_shoulder – right_elbow
    (7, 9),   # left_elbow – left_wrist
    (8, 10),  # right_elbow – right_wrist
    (5, 11),  # left_shoulder – left_hip
    (6, 12),  # right_shoulder – right_hip
    (11, 12), # left_hip – right_hip
    (11, 13), # left_hip – left_knee
    (12, 14), # right_hip – right_knee
    (13, 15), # left_knee – left_ankle
    (14, 16), # right_knee – right_ankle
]

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

        # Add batch dimension: (192, 192, 3) -> (1, 192, 192, 3)
        input_data = np.expand_dims(frame, axis=0).astype(np.uint8)

        output = client.infer(input_data)

        # Output shape is [1, 1, 17, 3] -> reshape to [17, 3]
        # Each row is [y, x, confidence] in normalized coordinates
        keypoints = np.array(output).reshape(17, 3)

        if i % 10 == 0:
            elapsed = time.monotonic() - start
            fps = (i + 1) / elapsed if elapsed > 0 else 0

            # Filter keypoints by confidence
            visible = [
                (idx, KEYPOINT_NAMES[idx], kp)
                for idx, kp in enumerate(keypoints)
                if kp[2] >= CONFIDENCE_THRESHOLD
            ]

            print(f"\nFrame {i} ({fps:.1f} FPS) — {len(visible)} keypoint(s):")
            for idx, name, kp in visible:
                print(f"  {name}: y={kp[0]:.3f} x={kp[1]:.3f} conf={kp[2]:.3f}")

            # Show skeleton connections where both endpoints are visible
            visible_ids = {idx for idx, _, _ in visible}
            connections = [
                (a, b) for a, b in SKELETON_CONNECTIONS
                if a in visible_ids and b in visible_ids
            ]
            if connections:
                print(f"  Skeleton ({len(connections)} connections):")
                for a, b in connections:
                    print(f"    {KEYPOINT_NAMES[a]} — {KEYPOINT_NAMES[b]}")

    elapsed = time.monotonic() - start
    print(f"\n{NUM_FRAMES} frames in {elapsed:.2f}s ({NUM_FRAMES / elapsed:.1f} FPS)")

picam2.stop()
