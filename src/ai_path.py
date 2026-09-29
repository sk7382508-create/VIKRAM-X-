"""Continuous-noise enhancement path using conservative spectral Wiener masking.

The interface remains a replaceable spectral-enhancement baseline (not a trained
neural network); the traffic director and fusion architecture are unchanged.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.preprocessing import compute_stft, compute_istft, DEFAULT_SR, HOP_LENGTH, FRAME_LENGTH


class AIContinuousEnhancementPath:
    """Suppress steady rotor/engine noise while retaining speech harmonics."""

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr
        self.dccrn_loaded = False
        self.dccrn_model = None

    def load_dccrn_weights(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False
        # Placeholder integration point; no DCCRN model is bundled in this demo.
        return False

    def process(
        self,
        noisy_audio: np.ndarray,
        prop_decrease: float = 0.94,
        noise_percentile: float = 45.0,
        subtraction_strength: float = 2.6,
        gain_floor: float = 0.05,
    ) -> np.ndarray:
        if len(noisy_audio) == 0:
            return noisy_audio.copy()

        spectrum = compute_stft(noisy_audio, win_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        magnitude = np.abs(spectrum)
        # A low temporal percentile estimates persistent interference without
        # mistaking the opening speech frame for a dedicated noise-only sample.
        noise = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)
        noise = gaussian_filter1d(noise, sigma=1.2, axis=0)
        # Spectral subtraction is more decisive against the persistent rotor
        # bed than a mild Wiener mask. A nonzero floor protects weak consonants.
        gain = np.maximum(gain_floor, 1.0 - (subtraction_strength * prop_decrease) * noise / (magnitude + 1e-10))
        enhanced = compute_istft(spectrum * gain, hop_length=HOP_LENGTH,
                                 win_length=FRAME_LENGTH, length=len(noisy_audio))
        return np.nan_to_num(enhanced).astype(np.float32)
