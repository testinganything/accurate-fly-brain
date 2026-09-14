#!/usr/bin/env python3
"""
Advanced MaleCNS whole-CNS simulation with live visual dashboard.

Feeds any video into the real fruit-fly visual system and shows:
  - Activity over time in key pathways (escape, descending, motor, visual)
  - 3D map of currently active neurons (if positions available)
  - Summary statistics and strongest-driven cell types
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from flybrain import FlyBrain
except ImportError:
    print("Please install the package first:  pip install -r requirements.txt")
    sys.exit(1)

from visual_encoder import VisualFrontEnd
from dashboard import Dashboard


# Key pathways we track (real MaleCNS cell types / superclasses)
PATHWAYS = {
    "looming / escape": ["LC4", "LPLC2", "DNp01"],
    "descending": ["descending_neuron"],
    "motor": ["motor_neuron", "MN"],
    "visual projection": ["visual_projection", "visual_projection_neuron"],
    "optic lobe": ["T4", "T5", "Tm", "Mi", "L1", "L2"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Advanced video → MaleCNS fly brain with visual dashboard"
    )
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds of video to process")
    parser.add_argument("--gain", type=float, default=0.9, help="Visual drive strength")
    parser.add_argument("--dt", type=float, default=0.020, help="Brain step (s)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-live", action="store_true", help="Skip live window (only save plots at end)")
    parser.add_argument("--out", type=str, default="output", help="Folder for saved plots")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MaleCNS brain...")
    brain = FlyBrain(device=args.device, dt=args.dt)
    print(f"Brain ready — {brain.n} neurons  |  device={brain.device}")

    # Resolve pathway neuron indices once
    pathway_idx = {}
    for name, types in PATHWAYS.items():
        try:
            idx = brain.cells(types)
            if len(idx) > 0:
                pathway_idx[name] = idx
                print(f"  tracked '{name}': {len(idx)} neurons")
        except Exception:
            pass

    # Also try Giant Fiber specifically
    try:
        gf = brain.cells(["DNp01"])
        if len(gf):
            pathway_idx["Giant Fiber (DNp01)"] = gf
            print(f"  tracked Giant Fiber: {len(gf)} neurons")
    except Exception:
        pass

    frontend = VisualFrontEnd(brain, gain=args.gain)
    dash = Dashboard(
        brain,
        pathway_idx,
        live=not args.no_live,
        title=f"MaleCNS ← {video_path.name}",
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(args.duration * fps)
    # Process roughly one frame per brain step for smoother drive
    frames_per_step = max(1, int(round(fps * args.dt)))

    print(f"\nProcessing up to {args.duration}s @ ~{fps:.1f} fps")
    print("Close the dashboard window or wait for finish to save final plots.\n")

    history = defaultdict(list)  # pathway → list of spike counts
    times = []
    total_spikes_hist = []

    step = 0
    frame_i = 0
    pbar = tqdm(total=total_frames, unit="frame")

    while frame_i < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Only step the brain every frames_per_step video frames
        if frame_i % frames_per_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 72))

            eye_drive, inject = frontend.encode(gray)
            fired = brain.step(eye_drive=eye_drive, inject=inject)

            t = step * args.dt
            times.append(t)
            n_active = len(fired) if fired is not None else 0
            total_spikes_hist.append(n_active)

            fired_set = set(fired.tolist()) if fired is not None and len(fired) else set()

            for name, idx in pathway_idx.items():
                count = sum(1 for i in idx if i in fired_set)
                history[name].append(count)

            # Live dashboard update
            dash.update(
                t=t,
                frame=frame,
                fired=fired,
                pathway_counts={k: history[k][-1] for k in history},
                total_active=n_active,
            )

            step += 1

        frame_i += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print("\nFinished processing. Saving plots...")
    dash.finalize(
        times=times,
        history=dict(history),
        total_spikes=total_spikes_hist,
        out_dir=out_dir,
    )
    print(f"Plots saved to: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
