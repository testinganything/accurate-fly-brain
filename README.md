# Accurate Fly Brain (MaleCNS)

Most practical accurate whole-CNS *Drosophila melanogaster* simulation currently available to the public.

- **Connectome**: MaleCNS v1.0 (HHMI Janelia / Cambridge / Google Research, 2026) — ~166,700 neurons, brain + ventral nerve cord, ~25.6 M directed connections in the usable graph.
- **Neuron model**: Leaky integrate-and-fire (LIF) following the exact parameters published in Shiu et al., *Nature* 2024 (the model that already predicted real fly taste → proboscis and grooming circuits at high accuracy).
- **Signs**: From official neurotransmitter predictions (ACh excitatory; GABA / glutamate / histamine inhibitory).
- **No learned policy inside the brain** — pure connectome dynamics.

This repo gives you a clean, installable wrapper + a video sensory front-end so you can drive the real visual system with any video file (games, movies, adult content, webcam, etc.).

## Quick start (recommended)

```bash
git clone https://github.com/testinganything/accurate-fly-brain.git
cd accurate-fly-brain
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# First run downloads the ~260 MB MaleCNS weight files automatically
python watch_video.py path/to/your/video.mp4
```

The script samples frames, converts them to drive on the fly’s visual projection / looming / motion neurons, steps the full CNS, and prints descending-neuron / motor activity.

## What “most accurate as possible” means here

| Component              | Choice                                      | Source / reason |
|------------------------|---------------------------------------------|-----------------|
| Wiring                 | MaleCNS v1.0 full CNS                       | Largest dense adult CNS map |
| Neuron dynamics        | Identical LIF for every cell                | Shiu et al. 2024 (validated on real behavior) |
| Rest / threshold       | –52 mV / –45 mV                             | Kakaria & de Bivort / Shiu |
| τ_m / τ_syn / refractory | 20 ms / 5 ms / 2.2 ms                     | Published values |
| Single-synapse weight  | 0.275 mV (or calibrated gain)               | Free parameter fitted in Shiu paper |
| Synapse sign           | NT prediction                               | Official MaleCNS annotations |

This is still a **point-neuron approximation**. It deliberately ignores morphology, detailed ion channels, neuromodulation dynamics, and individual variability. It is currently the best publicly runnable model that stays faithful to the published connectome + validated LIF parameters.

## Watching any video (including adult content)

```bash
python watch_video.py /path/to/video.mp4 --duration 30 --gain 1.0
```

- Frames → luminance + simple motion energy → injected into visual projection neurons / LC / LPLC groups (the same pathway used for looming and motion in the real fly).
- The brain is completely agnostic to the semantic content of the pixels. Porn, nature documentary, or Doom are just different spatiotemporal patterns driving the same photoreceptor / visual-projection pathway.
- Output: spike rates of key descending neurons and motor pools every few hundred ms.

See `watch_video.py` and `visual_encoder.py` for the exact mapping. You can change which cell types receive the drive.

## Installation notes

- Python ≥ 3.10
- CPU works (numba). GPU (CuPy / CUDA 12) is much faster if available.
- First run of the brain downloads the pre-built MaleCNS weight matrix (~260 MB) to `~/fly-data` (or `$FLY_DATA`).
- Data license: MaleCNS is CC-BY 4.0. Cite Berg et al., *Cell* 2026 and Shiu et al., *Nature* 2024.

## Citation

If you use this, please cite:

- Berg et al. (2026) Sexual dimorphism in the complete connectome of the Drosophila male central nervous system. *Cell*.
- Shiu et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634:210–219.

## Disclaimer

This is a scientific simulation of neural dynamics, not a conscious agent, not a living animal, and not an AI that “understands” content. It simply propagates spikes according to the real wiring diagram.
