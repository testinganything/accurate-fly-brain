"""
Live dashboard via OpenCV (works without Tcl/Tk) + matplotlib plots saved at the end.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Matplotlib only for final saved plots (Agg = no GUI needed)
HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    pass


class Dashboard:
    def __init__(self, brain, pathway_idx: dict, live: bool = True, title: str = "MaleCNS"):
        self.brain = brain
        self.pathway_idx = pathway_idx
        self.title = title
        self.positions = getattr(brain, "positions", None)
        self.live = live

        # Precompute valid (non-NaN) position bounds once
        self._xy = None
        self._mins = None
        self._span = None
        if self.positions is not None:
            pos = np.asarray(self.positions, dtype=np.float64)
            if pos.ndim == 2 and pos.shape[1] >= 2:
                xy = pos[:, :2]
                valid = np.isfinite(xy).all(axis=1)
                if valid.any():
                    self._xy = xy
                    self._valid_mask = valid
                    good = xy[valid]
                    self._mins = good.min(axis=0)
                    maxs = good.max(axis=0)
                    self._span = np.maximum(maxs - self._mins, 1e-6)

        self._t = []
        self._total = []
        self._path = {k: [] for k in pathway_idx}

        self.window = "MaleCNS Live Dashboard"
        if self.live:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, 1280, 720)
            print("[dashboard] Live OpenCV window opened (press Q in the window to stop early).")

    def update(self, t, frame, fired, pathway_counts, total_active):
        self._t.append(t)
        self._total.append(total_active)
        for k, v in pathway_counts.items():
            self._path[k].append(v)

        if not self.live:
            return

        canvas = self._render(frame, fired, pathway_counts, total_active, t)
        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.stop_requested = True

    @property
    def stop_requested(self):
        return getattr(self, "_stop", False)

    @stop_requested.setter
    def stop_requested(self, value):
        self._stop = bool(value)

    def _render(self, frame, fired, pathway_counts, total_active, t):
        """Build a single BGR image: video + stats + bars + neuron map."""
        W, H = 1280, 720
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # --- Left: video frame ---
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(620 / fw, 400 / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            y0 = 40
            x0 = 20
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
            cv2.putText(
                canvas,
                "Input frame",
                (x0, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # --- Top-right: big numbers ---
        cv2.putText(
            canvas,
            f"t = {t:6.2f} s",
            (680, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Active neurons: {total_active}",
            (680, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )

        # --- Pathway bars ---
        cv2.putText(
            canvas,
            "Pathway activity (this step)",
            (680, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        max_bar = 500
        y = 160
        colors = [
            (0, 165, 255),
            (0, 255, 128),
            (255, 180, 0),
            (255, 100, 255),
            (100, 200, 255),
            (180, 180, 80),
        ]
        for i, (name, count) in enumerate(pathway_counts.items()):
            color = colors[i % len(colors)]
            bar_w = int(min(max_bar, count * 0.08 + (20 if count else 0)))
            cv2.rectangle(canvas, (680, y), (680 + bar_w, y + 22), color, -1)
            label = f"{name}: {count}"
            cv2.putText(
                canvas,
                label,
                (680, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 32

        # --- Bottom: activity sparkline (total) ---
        spark_x0, spark_y0 = 20, 480
        spark_w, spark_h = 600, 200
        cv2.rectangle(
            canvas,
            (spark_x0, spark_y0),
            (spark_x0 + spark_w, spark_y0 + spark_h),
            (50, 50, 50),
            -1,
        )
        cv2.putText(
            canvas,
            "Total active neurons over time",
            (spark_x0, spark_y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if len(self._total) >= 2:
            vals = np.array(self._total[-300:], dtype=np.float32)
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax <= vmin:
                vmax = vmin + 1
            pts = []
            for i, v in enumerate(vals):
                x = spark_x0 + int(i / max(len(vals) - 1, 1) * (spark_w - 1))
                yy = spark_y0 + spark_h - 1 - int((v - vmin) / (vmax - vmin) * (spark_h - 1))
                pts.append((x, yy))
            for a, b in zip(pts, pts[1:]):
                cv2.line(canvas, a, b, (80, 80, 255), 2, cv2.LINE_AA)

        # --- Right bottom: neuron map ---
        map_x0, map_y0 = 680, 400
        map_w, map_h = 560, 280
        cv2.rectangle(
            canvas,
            (map_x0, map_y0),
            (map_x0 + map_w, map_y0 + map_h),
            (45, 45, 45),
            -1,
        )
        cv2.putText(
            canvas,
            "Active somata (X-Y)",
            (map_x0, map_y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        drawn = False
        if (
            self._xy is not None
            and self._mins is not None
            and fired is not None
            and len(fired)
        ):
            idx = np.asarray(fired, dtype=np.int64)
            # Keep only indices that exist and have finite coords
            idx = idx[(idx >= 0) & (idx < len(self._xy))]
            if len(idx):
                if len(idx) > 4000:
                    idx = np.random.choice(idx, 4000, replace=False)
                pts = self._xy[idx]
                finite = np.isfinite(pts).all(axis=1)
                pts = pts[finite]
                if len(pts):
                    norm = (pts - self._mins) / self._span
                    # Clip to [0,1] just in case
                    norm = np.clip(norm, 0.0, 1.0)
                    for p in norm:
                        x = map_x0 + int(p[0] * (map_w - 1))
                        yy = map_y0 + int((1.0 - p[1]) * (map_h - 1))
                        cv2.circle(canvas, (x, yy), 1, (60, 60, 255), -1)
                    drawn = True

        if not drawn:
            cv2.putText(
                canvas,
                "No positions or no spikes",
                (map_x0 + 40, map_y0 + map_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Press Q in this window to stop early",
            (20, H - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def finalize(self, times, history, total_spikes, out_dir: Path):
        if self.live:
            try:
                cv2.destroyWindow(self.window)
            except Exception:
                pass

        if not HAS_MPL:
            print("matplotlib not available — skipping PNG export")
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(self.title)

        axes[0].plot(times, total_spikes, color="#e74c3c", lw=1.5)
        axes[0].set_ylabel("Total active neurons")
        axes[0].set_title("Whole-CNS activity")
        axes[0].grid(True, alpha=0.3)

        for name, series in history.items():
            if len(series) == len(times):
                axes[1].plot(times, series, label=name, lw=1.3)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("# spiking in pathway")
        axes[1].set_title("Key pathways")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        p1 = out_dir / "activity_over_time.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"  saved {p1}")

        if history:
            names = list(history.keys())
            means = [float(np.mean(history[n])) if history[n] else 0.0 for n in names]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(names, means, color="#3498db")
            ax.set_xlabel("Mean # spiking neurons per step")
            ax.set_title("Average pathway drive")
            fig.tight_layout()
            p2 = out_dir / "pathway_summary.png"
            fig.savefig(p2, dpi=150)
            plt.close(fig)
            print(f"  saved {p2}")
