#!/usr/bin/env python3
"""
Drive the MaleCNS whole-CNS LIF model with frames from any video file.

Works with ordinary videos, games, nature footage, or adult content —
the brain only sees spatiotemporal luminance / motion patterns mapped onto
its real visual projection neurons.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from flybrain import FlyBrain
except ImportError:
    print("Please install the package first:  pip install -r requirements.txt")
    sys.exit(1)

from visual_encoder import frames_to_visual_drive


def main():
    parser = argparse.ArgumentParser(description="Feed any video into the MaleCNS fly brain")
    parser.add_argument("video", type=str, help="Path to video file (mp4, avi, mkv, ...)")
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds of video to process")
    parser.add_argument("--gain", type=float, default=0.8, help="Overall visual drive strength")
    parser.add_argument("--dt", type=float, default=0.020, help="Brain integration step (s)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print("Loading MaleCNS brain (first run downloads ~260 MB weights)...")
    brain = FlyBrain(device=args.device)
    # FlyBrain exposes the neuron count as .n (not .n_neurons)
    n_neurons = getattr(brain, "n", getattr(brain, "n_neurons", "?"))
    print(f"Brain ready — {n_neurons} neurons")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(args.duration * fps)

    print(f"Processing up to {args.duration}s of video @ ~{fps:.1f} fps")
    print("Mapping frames → visual projection / looming neurons → full CNS dynamics")
    print("-" * 60)

    step = 0
    pbar = tqdm(total=total_frames, unit="frame")
    while step < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → grayscale luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 48))  # modest resolution is enough for the mapping

        # Turn the image into current injection on real visual cell types
        inject = frames_to_visual_drive(brain, gray, gain=args.gain)

        fired = brain.step(inject=inject)

        # Simple readout every ~0.5 s of brain time
        if step % max(1, int(0.5 / args.dt)) == 0:
            n_spikes = len(fired) if fired is not None else 0
            print(f"t={step * args.dt:6.2f}s | active neurons this step: {n_spikes}")

        step += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print("Done. The brain processed the visual stream through its real wiring.")


if __name__ == "__main__":
    main()
