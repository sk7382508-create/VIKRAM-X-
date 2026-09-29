"""
VIKRAM-X: Audio Reconstruction and Post-Processing Module
Handles DC bias removal, communication speech level calibration,
and final WAV export to outputs/enhanced_battlefield.wav.
"""

import os
import numpy as np
import soundfile as sf
from src.preprocessing import DEFAULT_SR


def reconstruct_and_export(
    audio_signal: np.ndarray,
    output_path: str = "outputs/enhanced_battlefield.wav",
    sr: int = DEFAULT_SR,
    target_peak: float = 0.88,
) -> tuple[str, np.ndarray]:
    """
    Finalize enhanced communication signal and export to WAV.
    - Removes DC bias offset
    - Preserves output level and clips only peaks above the safe target
    - Writes standard 16-bit PCM WAV playable in browser / Streamlit
    """
    if len(audio_signal) == 0:
        raise ValueError("Audio signal is empty")

    clean = audio_signal.astype(np.float32)
    # Remove DC bias
    clean = clean - np.mean(clean)

    # Preserve the processed signal's level. Peak-normalizing residual noise can
    # make an apparently enhanced recording louder and noisier than the input.
    peak = float(np.max(np.abs(clean)))
    if peak > target_peak:
        clean *= target_peak / peak

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, clean, sr, subtype="PCM_16")

    return output_path, clean
