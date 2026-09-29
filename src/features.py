"""
VIKRAM-X: Audio Feature Extraction Module
Extracts acoustic and spectral features per 20 ms analysis frame:
RMS, Crest Factor, Kurtosis, Spectral Flux, Spectral Centroid,
Zero-Crossing Rate (ZCR), Sub-band energies, and Estimated SNR.
"""

import numpy as np
import scipy.stats
from src.preprocessing import compute_stft, FRAME_LENGTH, HOP_LENGTH, DEFAULT_SR, N_FFT


def extract_frame_features(
    signal: np.ndarray,
    sr: int = DEFAULT_SR,
    frame_len: int = FRAME_LENGTH,
    hop_len: int = HOP_LENGTH,
) -> dict:
    """
    Extract comprehensive temporal and spectral features across all frames.
    
    Returns dictionary with 1D numpy arrays of length n_frames:
    - times: timestamp of each frame (seconds)
    - rms: root-mean-square amplitude
    - crest_factor: ratio of peak to RMS (linear)
    - crest_factor_db: crest factor in decibels
    - kurtosis: 4th standardized statistical moment (high for shockwaves)
    - spectral_flux: frame-to-frame spectral variation
    - spectral_centroid: frequency center of gravity (Hz)
    - zcr: zero-crossing rate
    - low_band_energy: power in 0 - 500 Hz (helicopter rotor dominance)
    - mid_band_energy: power in 500 - 3000 Hz (speech formant band)
    - high_band_energy: power in 3000 - 8000 Hz (transient crack / air turbulence)
    - estimated_snr: running estimated instantaneous SNR (dB)
    - stft_mag: STFT magnitude matrix (n_bins, n_frames)
    - stft_complex: STFT complex matrix (n_bins, n_frames)
    """
    # 1. Compute STFT
    stft_complex = compute_stft(signal, hop_length=hop_len, win_length=frame_len)
    stft_mag = np.abs(stft_complex)
    n_bins, n_frames = stft_mag.shape
    times = np.arange(n_frames, dtype=np.float32) * hop_len / sr

    # 2. Temporal frame extraction (with centered padding matching STFT)
    pad_len = frame_len // 2
    padded_signal = np.pad(signal, (pad_len, pad_len), mode="reflect")

    rms_arr = np.zeros(n_frames, dtype=np.float32)
    crest_arr = np.zeros(n_frames, dtype=np.float32)
    crest_db_arr = np.zeros(n_frames, dtype=np.float32)
    kurtosis_arr = np.zeros(n_frames, dtype=np.float32)
    zcr_arr = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = padded_signal[start:end]

        # RMS
        r = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        rms_arr[i] = r

        # Crest Factor
        peak = float(np.max(np.abs(frame)))
        cf = peak / (r + 1e-9)
        crest_arr[i] = cf
        crest_db_arr[i] = float(20.0 * np.log10(max(1.0, cf)))

        # Kurtosis (Fisher kurtosis: normal distribution = 0.0)
        # Using unbiased moment calculation
        var = float(np.var(frame))
        if var > 1e-8:
            m4 = float(np.mean((frame - np.mean(frame)) ** 4))
            kurt = (m4 / (var ** 2)) - 3.0
            kurtosis_arr[i] = float(max(-3.0, min(150.0, kurt)))
        else:
            kurtosis_arr[i] = 0.0

        # Zero Crossing Rate
        crossings = np.sum(np.abs(np.diff(np.sign(frame) >= 0)))
        zcr_arr[i] = float(crossings / (len(frame) - 1))

    # 3. Spectral features
    # Spectral Flux: Euclidean difference between successive magnitude frames
    flux = np.zeros(n_frames, dtype=np.float32)
    diff = np.diff(stft_mag, axis=1)
    # Rectified spectral flux (half-wave rectified to highlight onset energy surges)
    pos_diff = np.maximum(0.0, diff)
    flux[1:] = np.sqrt(np.mean(pos_diff ** 2, axis=0))

    # Spectral Centroid
    freq_bins = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    spec_sum = np.sum(stft_mag, axis=0) + 1e-9
    centroid = np.sum(stft_mag * freq_bins[:, np.newaxis], axis=0) / spec_sum

    # Sub-band Energies
    idx_low = freq_bins <= 500.0
    idx_mid = (freq_bins > 500.0) & (freq_bins <= 3000.0)
    idx_high = freq_bins > 3000.0

    low_band = np.mean(stft_mag[idx_low, :] ** 2, axis=0)
    mid_band = np.mean(stft_mag[idx_mid, :] ** 2, axis=0)
    high_band = np.mean(stft_mag[idx_high, :] ** 2, axis=0)

    # 4. Instantaneous Frame SNR Estimation
    # Robust percentile-based noise floor tracking over sliding 1-second window
    window_frames = int(1.0 * sr / hop_len)
    total_energy = np.mean(stft_mag ** 2, axis=0)
    
    # Running 15th percentile energy as noise floor estimate
    half_w = window_frames // 2
    est_noise_floor = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        w_start = max(0, i - half_w)
        w_end = min(n_frames, i + half_w)
        est_noise_floor[i] = float(np.percentile(total_energy[w_start:w_end], 15))

    ratio = np.maximum(1.05, total_energy / est_noise_floor)
    snr_est_arr = 10.0 * np.log10(ratio - 1.0)
    snr_est_arr = np.clip(np.nan_to_num(snr_est_arr, nan=-20.0), -20.0, 30.0)

    return {
        "times": times,
        "rms": rms_arr,
        "crest_factor": crest_arr,
        "crest_factor_db": crest_db_arr,
        "kurtosis": kurtosis_arr,
        "spectral_flux": flux,
        "spectral_centroid": centroid.astype(np.float32),
        "zcr": zcr_arr,
        "low_band_energy": low_band.astype(np.float32),
        "mid_band_energy": mid_band.astype(np.float32),
        "high_band_energy": high_band.astype(np.float32),
        "estimated_snr": snr_est_arr.astype(np.float32),
        "stft_mag": stft_mag,
        "stft_complex": stft_complex,
        "n_frames": n_frames,
    }
