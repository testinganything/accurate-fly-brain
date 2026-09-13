"""
Simple but biologically-motivated visual front-end.

Maps a grayscale frame onto groups of visual projection neurons / looming detectors
that actually exist in the MaleCNS connectome (LC4, LPLC2, HS/VS-like, etc.).

This is deliberately lightweight. More sophisticated optic-flow or columnar
models can be swapped in later; the important part is that the *targets* are
real cell types from the connectome.
"""

from __future__ import annotations

import numpy as np


def frames_to_visual_drive(brain, gray: np.ndarray, gain: float = 0.8):
    """
    gray : 2-D uint8 or float array (H, W), luminance
    Returns a list of (neuron_ids, current) pairs suitable for brain.step(inject=...)
    """
    # Normalise to 0–1
    img = gray.astype(np.float32) / 255.0

    # Very simple features that map onto known fly visual channels
    mean_lum = float(img.mean())
    # Crude motion proxy: difference from a heavily blurred version
    from scipy.ndimage import gaussian_filter
    blur = gaussian_filter(img, sigma=3.0)
    motion = float(np.abs(img - blur).mean())

    # Centre-surround contrast (rough looming / object signal)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    centre = float(img[mask].mean()) if mask.any() else mean_lum
    surround = float(img[~mask].mean()) if (~mask).any() else mean_lum
    contrast = abs(centre - surround)

    inject = []

    # Map onto real cell-type groups that exist in MaleCNS annotations.
    # These names are the ones used by the flybrain package / MaleCNS cell types.
    # Adjust the list if the package exposes different labels.

    try:
        # Looming / expansion detectors (classic escape pathway)
        loom_cells = brain.cells(["LC4", "LPLC2"], side=None)  # both sides if available
        if len(loom_cells) > 0:
            strength = gain * (0.4 * motion + 0.6 * contrast)
            inject.append((loom_cells, strength))

        # Broad visual projection / motion-sensitive groups (approximate)
        # Many packages expose broader super-classes; fall back gracefully.
        for name in ["visual_projection_neuron", "VPNs", "HS", "VS", "T4", "T5"]:
            try:
                cells = brain.cells([name], side=None)
                if len(cells) > 0:
                    inject.append((cells, gain * 0.3 * mean_lum))
                    break
            except Exception:
                continue

    except Exception as e:
        # If the exact cell-type lookup fails, inject a small random set of
        # sensory-looking neurons so the demo still runs.
        print(f"[visual_encoder] cell lookup note: {e}")
        # Fallback: mild drive on a handful of neurons so something happens
        n = min(200, brain.n_neurons)
        inject.append((np.arange(n), gain * 0.1 * mean_lum))

    return inject
