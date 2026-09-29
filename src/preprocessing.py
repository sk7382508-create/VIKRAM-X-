"""
VIKRAM-X: Audio Preprocessing and Time-Frequency Analysis Module
Implements 16 kHz mono framing (20 ms window, 10 ms hop),
STFT analysis, and COLA-compliant iSTFT synthesis.
"""

import numpy as np
import scipy.signal


DEFAULT_SR = 16000
FRAME_LENGTH = 320   # 20 ms at 16 kHz
HOP_LENGTH = 160     # 10 ms at 16 kHz
N_FFT = 512          # Frequency resolution (257 bins from 0 to 8000 Hz)


def frame_signal(signal: np.ndarray, frame_len: int = FRAME_LENGTH, hop_len: int = HOP_LENGTH) -> np.ndarray:
    """
    Segment 1D signal into overlapping frames.
    Returns 2D array of shape (n_frames, frame_len).
    """
    n_samples = len(signal)
    if n_samples < frame_len:
        padded = np.pad(signal, (0, frame_len - n_samples))
        return padded[np.newaxis, :]

    n_frames = 1 + (n_samples - frame_len) // hop_len
    shape = (n_frames, frame_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return frames.copy()


def compute_stft(
    signal: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = FRAME_LENGTH,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform (STFT) using Hann window.
    Returns complex spectrogram of shape (1 + n_fft//2, n_frames).
    """
    _, _, z = scipy.signal.stft(
        signal.astype(np.float32), fs=1.0, window="hann", nperseg=win_length,
        noverlap=win_length-hop_length, nfft=n_fft, boundary="zeros", padded=True,
    )
    return z


def compute_istft(
    stft_matrix: np.ndarray,
    hop_length: int = HOP_LENGTH,
    win_length: int = FRAME_LENGTH,
    length: int | None = None,
) -> np.ndarray:
    """
    Compute inverse Short-Time Fourier Transform (iSTFT) to reconstruct time domain audio.
    COLA compliant with symmetric Hann window.
    """
    _, reconstructed = scipy.signal.istft(
        stft_matrix, fs=1.0, window="hann", nperseg=win_length,
        noverlap=win_length-hop_length, nfft=(stft_matrix.shape[0]-1)*2,
        input_onesided=True, boundary=True,
    )
    if length is not None:
        reconstructed = reconstructed[:length]
        if len(reconstructed) < length:
            reconstructed = np.pad(reconstructed, (0, length-len(reconstructed)))
    return reconstructed.astype(np.float32)


def get_frame_times(n_frames: int, hop_length: int = HOP_LENGTH, sr: int = DEFAULT_SR) -> np.ndarray:
    """Get time in seconds for the center of each STFT/analysis frame."""
    return np.arange(n_frames, dtype=np.float32) * hop_length / sr
