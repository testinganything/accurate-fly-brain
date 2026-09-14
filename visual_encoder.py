"""
Improved visual front-end for MaleCNS.

Uses the package's native eye_drive (photoreceptor array) when available,
plus targeted current injection into looming / visual-projection cell types.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class VisualFrontEnd:
    def __init__(self, brain, gain: float = 0.9):
        self.brain = brain
        self.gain = gain

        # Photoreceptor indices exposed by flybrain
        self.n_visual = len(brain.visual) if hasattr(brain, "visual") else 0

        # Pre-resolve important visual cell groups
        self.loom = self._safe_cells(["LC4", "LPLC2"])
        self.vp = self._safe_cells(["visual_projection", "visual_projection_neuron"])
        self.motion = self._safe_cells(["T4", "T5", "HS", "VS"])

        # Keep a previous frame for simple motion energy
        self.prev = None

    def _safe_cells(self, types):
        try:
            idx = self.brain.cells(types)
            return idx if len(idx) > 0 else np.array([], dtype=int)
        except Exception:
            return np.array([], dtype=int)

    def encode(self, gray: np.ndarray):
        """
        gray : (H, W) uint8 or float luminance frame
        Returns (eye_drive, inject)
          eye_drive : array for brain.step(eye_drive=...) or None
          inject    : list of (indices, amount) for brain.step(inject=...)
        """
        img = gray.astype(np.float32) / 255.0
        mean_lum = float(img.mean())

        # Motion energy
        if self.prev is not None:
            motion = float(np.abs(img - self.prev).mean())
        else:
            motion = 0.0
        self.prev = img.copy()

        # Centre-surround contrast (object / looming proxy)
        h, w = img.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        centre = float(img[mask].mean()) if mask.any() else mean_lum
        surround = float(img[~mask].mean()) if (~mask).any() else mean_lum
        contrast = abs(centre - surround)

        # --- eye_drive (native photoreceptor path) ---
        eye_drive = None
        if self.n_visual > 0:
            # Simple retinotopic-ish mapping: resize image to a 1-D drive
            # of length n_visual. Real columnar mapping is more complex;
            # this still gives spatially structured input.
            flat = cv2_resize_1d(img, self.n_visual)
            eye_drive = np.clip(flat * self.gain, 0.0, 1.0).astype(np.float32)

        # --- targeted inject into known visual cell types ---
        inject = []
        loom_strength = self.gain * (0.55 * motion + 0.45 * contrast)
        if len(self.loom):
            inject.append((self.loom, loom_strength))

        if len(self.vp):
            inject.append((self.vp, self.gain * 0.25 * mean_lum))

        if len(self.motion):
            inject.append((self.motion, self.gain * 0.35 * motion))

        return eye_drive, inject


def cv2_resize_1d(img: np.ndarray, n: int) -> np.ndarray:
    """Resize 2-D image to a length-n vector (row-major after modest resize)."""
    import cv2
    # Keep aspect, then flatten and resample to exactly n values
    h, w = img.shape
    target_h = max(8, int(np.sqrt(n * h / w)))
    target_w = max(8, n // target_h)
    small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    flat = small.ravel()
    if len(flat) == n:
        return flat
    # Linear resample
    x_old = np.linspace(0, 1, len(flat))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, flat).astype(np.float32)
