"""
Live + final visual dashboard for the MaleCNS video simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")  # interactive on Windows
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class Dashboard:
    def __init__(self, brain, pathway_idx: dict, live: bool = True, title: str = "MaleCNS"):
        self.brain = brain
        self.pathway_idx = pathway_idx
        self.live = live and HAS_MPL
        self.title = title
        self.positions = getattr(brain, "positions", None)

        self.fig = None
        self.axes = {}
        self._pathway_lines = {}
        self._total_line = None
        self._scatter = None
        self._img_artist = None
        self._text = None

        # Rolling history for live plot
        self._t = []
        self._total = []
        self._path = {k: [] for k in pathway_idx}

        if self.live:
            self._init_live()

    def _init_live(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(self.title, fontsize=13)
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.1, 1], width_ratios=[1.1, 1, 1])

        # Video frame
        self.axes["frame"] = self.fig.add_subplot(gs[0, 0])
        self.axes["frame"].set_title("Input frame")
        self.axes["frame"].axis("off")

        # Total activity
        self.axes["total"] = self.fig.add_subplot(gs[0, 1])
        self.axes["total"].set_title("Total active neurons")
        self.axes["total"].set_xlabel("Time (s)")
        self.axes["total"].set_ylabel("# spiking")
        (self._total_line,) = self.axes["total"].plot([], [], color="#e74c3c", lw=1.5)

        # Pathway activity
        self.axes["path"] = self.fig.add_subplot(gs[0, 2])
        self.axes["path"].set_title("Key pathways")
        self.axes["path"].set_xlabel("Time (s)")
        self.axes["path"].set_ylabel("# spiking")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.pathway_idx), 1)))
        for (name, _), c in zip(self.pathway_idx.items(), colors):
            (line,) = self.axes["path"].plot([], [], label=name, lw=1.4, color=c)
            self._pathway_lines[name] = line
        self.axes["path"].legend(loc="upper right", fontsize=7)

        # 3D / 2D neuron map
        self.axes["map"] = self.fig.add_subplot(gs[1, :])
        self.axes["map"].set_title("Active neurons (soma positions)")
        self.axes["map"].set_xlabel("X")
        self.axes["map"].set_ylabel("Y")
        if self.positions is not None:
            # Background: all somata faint
            pos = np.asarray(self.positions)
            self.axes["map"].scatter(pos[:, 0], pos[:, 1], s=1, c="#dddddd", alpha=0.4)
            self._scatter = self.axes["map"].scatter([], [], s=6, c="#e74c3c", alpha=0.85)
        else:
            self.axes["map"].text(0.5, 0.5, "No soma positions in this brain build",
                                  ha="center", va="center", transform=self.axes["map"].transAxes)

        self._text = self.fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
        plt.tight_layout()
        plt.show(block=False)

    def update(self, t, frame, fired, pathway_counts, total_active):
        if not self.live or self.fig is None:
            return

        self._t.append(t)
        self._total.append(total_active)
        for k, v in pathway_counts.items():
            self._path[k].append(v)

        # Frame
        if frame is not None:
            rgb = frame[:, :, ::-1]  # BGR → RGB
            if self._img_artist is None:
                self._img_artist = self.axes["frame"].imshow(rgb)
            else:
                self._img_artist.set_data(rgb)

        # Total line
        self._total_line.set_data(self._t, self._total)
        self.axes["total"].relim()
        self.axes["total"].autoscale_view()

        # Pathway lines
        for name, line in self._pathway_lines.items():
            line.set_data(self._t, self._path[name])
        self.axes["path"].relim()
        self.axes["path"].autoscale_view()

        # Active neuron map
        if self._scatter is not None and fired is not None and len(fired) and self.positions is not None:
            pos = np.asarray(self.positions)
            # Subsample for speed if huge
            idx = fired if len(fired) < 8000 else np.random.choice(fired, 8000, replace=False)
            pts = pos[idx]
            self._scatter.set_offsets(pts[:, :2])

        self._text.set_text(
            f"t = {t:6.2f}s   |   total active = {total_active:6d}   |   "
            + "  ".join(f"{k}: {v}" for k, v in pathway_counts.items())
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finalize(self, times, history, total_spikes, out_dir: Path):
        """Save high-quality static plots at the end."""
        if not HAS_MPL:
            print("matplotlib not available — skipping plot save")
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Activity over time
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(self.title)

        axes[0].plot(times, total_spikes, color="#e74c3c", lw=1.5)
        axes[0].set_ylabel("Total active neurons")
        axes[0].set_title("Whole-CNS activity")
        axes[0].grid(True, alpha=0.3)

        for name, series in history.items():
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

        # 2) Summary bar chart of mean pathway activity
        if history:
            names = list(history.keys())
            means = [np.mean(history[n]) for n in names]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(names, means, color="#3498db")
            ax.set_xlabel("Mean # spiking neurons per step")
            ax.set_title("Average pathway drive")
            fig.tight_layout()
            p2 = out_dir / "pathway_summary.png"
            fig.savefig(p2, dpi=150)
            plt.close(fig)
            print(f"  saved {p2}")

        # Keep live window open a moment so user can look
        if self.live and self.fig is not None:
            print("Close the live dashboard window to exit.")
            plt.ioff()
            plt.show()
