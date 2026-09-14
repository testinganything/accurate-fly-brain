# Accurate Fly Brain (MaleCNS) — Advanced

Whole-CNS *Drosophila* simulation (MaleCNS v1.0, ~166,700 neurons) with a **visual dashboard**.

Feed any video into the real fly visual system and watch:

- Live input frame
- Total network activity over time
- Activity in key pathways (looming/escape, descending neurons, motor, visual projection…)
- 3D/2D map of currently spiking neurons (soma positions)
- Saved high-resolution plots at the end

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\vaksa\scripts\accurate-fly-brain
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python watch_video.py "C:\path\to\your\video.mp4" --duration 30
```

A live window will open. When the run finishes, plots are written to the `output/` folder.

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--duration` | 20 | Seconds of video to process |
| `--gain` | 0.9 | Visual drive strength |
| `--device` | auto | `cpu` / `cuda` / `auto` |
| `--no-live` | off | Skip live window, only save plots |
| `--out` | output | Folder for saved PNGs |

Example:
```powershell
python watch_video.py ".\HOT MILF MILKS ME - Jennifer White.mp4" --duration 30 --gain 1.0
```

## What you get

1. **Live dashboard** (matplotlib window)
2. `output/activity_over_time.png` — total + pathway activity curves
3. `output/pathway_summary.png` — bar chart of average drive per pathway

## Model

- Connectome: MaleCNS v1.0 (brain + VNC)
- Dynamics: leaky integrate-and-fire (flybrain / Shiu-style parameters)
- Visual input: native `eye_drive` + targeted injection into LC4/LPLC2 and visual-projection neurons

Cite Berg et al. *Cell* 2026 and Shiu et al. *Nature* 2024 if you use this scientifically.
