"""
VIKRAM-X: Audio Management and Battlefield Acoustic Synthesis Module
Handles audio I/O, calibrated SNR mixing, and acoustic synthesis of
helicopter rotor noise, engine/wind turbulence, and gunfire shockwave impulses.
"""

import os
import numpy as np
import scipy.signal
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, and resample to target_sr.
    Returns normalized float32 array in [-1.0, 1.0].
    """
    y, sr = sf.read(file_path, dtype="float32", always_2d=True)
    y = np.mean(y, axis=1, dtype=np.float32)
    if sr != target_sr:
        divisor = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // divisor, sr // divisor).astype(np.float32)
    # Preserve recorded levels. Independent peak normalization changes the
    # speech/noise balance and makes before/after comparisons misleading.
    return np.nan_to_num(y).astype(np.float32), target_sr


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000) -> str:
    """
    Save floating point audio array to 16-bit PCM WAV.
    Clips safely to [-1.0, 1.0] to prevent wrap-around distortion.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(file_path, clipped, sr, subtype="PCM_16")
    return file_path


def calculate_power(signal: np.ndarray) -> float:
    """Calculate mean power of signal."""
    if len(signal) == 0:
        return 0.0
    return float(np.mean(signal.astype(np.float64) ** 2))


def calculate_rms(signal: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) amplitude."""
    return float(np.sqrt(max(1e-12, calculate_power(signal))))


def synthesize_helicopter_noise(duration: float, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """
    Acoustically synthesize authentic heavy attack helicopter / rotor noise.
    Components:
    1. Main rotor blade-passing frequency (BPF) fundamental (~18 Hz) + harmonics.
    2. Cyclic pitch aerodynamic blade-vortex interaction (BVI) amplitude modulation.
    3. Turbulent downwash turbulent broadband noise (bandpassed 40 - 750 Hz).
    4. Gas turbine engine compressor whine (~1850 Hz + harmonic sidebands).
    5. Tail rotor high-frequency blade slap (~90 Hz).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Main rotor blade slap fundamental and harmonics
    bpf = 18.5  # Hz (typical for heavy utility/attack helicopter)
    rotor_harmonics = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 12):
        weight = 1.0 / (h ** 0.85)
        phase = rng.uniform(0, 2 * np.pi)
        rotor_harmonics += (weight * np.sin(2 * np.pi * (bpf * h) * t + phase)).astype(np.float32)

    # 2. Cyclic blade slap modulation (sharp periodic pressure pulses)
    pulse_train = np.maximum(0.0, np.sin(2 * np.pi * bpf * t)) ** 4
    rotor_sound = rotor_harmonics * (0.6 + 0.8 * pulse_train)

    # 3. Turbulent rotor downwash (filtered colored noise)
    white = rng.standard_normal(n_samples).astype(np.float32)
    sos_rotor = scipy.signal.butter(4, [40, 750], btype="bandpass", fs=sr, output="sos")
    turbulent_downwash = scipy.signal.sosfilt(sos_rotor, white) * (0.8 + 0.6 * pulse_train)

    # 4. Tail rotor (higher pitch ~92.5 Hz)
    tail_bpf = 92.5
    tail_rotor = np.zeros(n_samples, dtype=np.float32)
    for h in range(1, 6):
        tail_rotor += (0.4 / h) * np.sin(2 * np.pi * (tail_bpf * h) * t + rng.uniform(0, np.pi))

    # 5. Turboshaft turbine whine
    turbine_freq = 1820.0
    turbine = 0.25 * np.sin(2 * np.pi * turbine_freq * t) + 0.12 * np.sin(2 * np.pi * (turbine_freq * 2) * t)

    # Combine all components
    composite = 0.55 * rotor_sound + 0.50 * turbulent_downwash + 0.25 * tail_rotor + 0.15 * turbine
    composite = composite / np.max(np.abs(composite))
    return composite.astype(np.float32)


def synthesize_engine_wind(duration: float, sr: int = 16000, seed: int = 101) -> np.ndarray:
    """
    Synthesize continuous armored vehicle engine rumble and aerodynamic wind rush.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    white = rng.standard_normal(n_samples).astype(np.float32)
    # Low-pass filter for rumble
    sos_rumble = scipy.signal.butter(3, 160, btype="lowpass", fs=sr, output="sos")
    rumble = scipy.signal.sosfilt(sos_rumble, white)

    # Wind noise with slow gust modulation
    sos_wind = scipy.signal.butter(2, [80, 500], btype="bandpass", fs=sr, output="sos")
    wind = scipy.signal.sosfilt(sos_wind, rng.standard_normal(n_samples).astype(np.float32))
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    wind = wind * gust

    composite = 0.7 * rumble + 0.5 * wind
    composite = composite / (np.max(np.abs(composite)) + 1e-6)
    return composite.astype(np.float32)


def synthesize_gunfire_impulse(duration: float = 0.12, sr: int = 16000, seed: int = 999) -> np.ndarray:
    """
    Synthesize realistic supersonic blast shockwave (Friedlander waveform) + ballistic crack.
    Characteristics:
    - Microsecond rise time (< 0.15 ms shock front)
    - High positive pressure peak
    - Exponential decay with negative rarefaction phase
    - Ground reflection bounce delay (~12-18 ms)
    - Extreme crest factor (> 18 dB) and high kurtosis (> 30)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Friedlander shockwave blast parameter
    t_pos = 0.0035  # positive phase duration (~3.5 ms)
    blast = np.zeros(n_samples, dtype=np.float32)

    # Shockwave front
    idx_blast = t < 0.05
    t_b = t[idx_blast]
    # P(t) = P0 * (1 - t/t_pos) * exp(-b * t / t_pos)
    b = 1.4
    p_wave = (1.0 - (t_b / t_pos)) * np.exp(-b * (t_b / t_pos))
    blast[idx_blast] = p_wave

    # Ballistic crack / muzzle resonance (high frequency ringing burst)
    burst_t = np.linspace(0, 0.015, int(0.015 * sr), endpoint=False)
    burst_env = np.exp(-burst_t / 0.002)
    burst_sig = burst_env * np.sin(2 * np.pi * 2400.0 * burst_t)
    blast[:len(burst_sig)] += 0.45 * burst_sig

    # Ground reflection echo (delayed ~14 ms, inverted phase, attenuated)
    delay_samples = int(0.014 * sr)
    if delay_samples < n_samples:
        echo_len = min(len(blast) - delay_samples, int(0.04 * sr))
        blast[delay_samples:delay_samples + echo_len] += -0.32 * blast[:echo_len]

    # Normalize peak to 1.0
    blast = blast / np.max(np.abs(blast))
    return blast.astype(np.float32)


def mix_battlefield_scenario(
    clean_speech: np.ndarray,
    snr_db: float = -5.0,
    include_engine: bool = True,
    impulse_times: list[float] | None = None,
    sr: int = 16000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Create a controlled battlefield noisy mixture adhering strictly to:
    1. Speech: reference level
    2. Continuous helicopter: calibrated to target SNR (e.g. -5 dB, -10 dB, 0 dB)
    3. Engine/wind: introduced during 3-5 s
    4. Gunfire: short high-energy impulses with peak amplitude approx +6 dB relative to speech peak
    
    Returns:
    (noisy_mixture, clean_speech, total_noise, metadata)
    """
    n_samples = len(clean_speech)
    duration = n_samples / sr
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Clean speech reference
    p_speech = calculate_power(clean_speech)
    peak_speech = float(np.max(np.abs(clean_speech)))

    # 2. Helicopter continuous background
    heli_raw = synthesize_helicopter_noise(duration, sr=sr)
    p_heli_raw = calculate_power(heli_raw)

    # Calculate target continuous noise power:
    # SNR = 10 * log10(P_speech / P_noise) -> P_target = P_speech / (10^(SNR/10))
    target_noise_power = p_speech / (10.0 ** (snr_db / 10.0))
    scale_heli = np.sqrt(target_noise_power / (p_heli_raw + 1e-12))
    heli_continuous = (heli_raw * scale_heli).astype(np.float32)

    total_continuous_noise = heli_continuous.copy()

    # 3. Optional Engine/wind background between 3s and 5s (and ongoing)
    if include_engine:
        engine_raw = synthesize_engine_wind(duration, sr=sr)
        # Smooth window for engine entry at 3s - 5s
        engine_envelope = np.zeros(n_samples, dtype=np.float32)
        idx_engine = t >= 3.0
        # Smooth ramp in from 3.0s to 3.5s
        ramp_len = int(0.5 * sr)
        start_idx = int(3.0 * sr)
        if start_idx < n_samples:
            actual_ramp = min(ramp_len, n_samples - start_idx)
            engine_envelope[start_idx:start_idx + actual_ramp] = np.linspace(0, 1, actual_ramp)
            engine_envelope[start_idx + actual_ramp:] = 1.0

        scale_engine = 0.45 * scale_heli
        total_continuous_noise += (engine_raw * scale_engine * engine_envelope).astype(np.float32)

    # 4. Gunfire impulsive events at specified timestamps (default: 5.0s and 8.0s)
    if impulse_times is None:
        impulse_times = [5.0, 8.0]

    impulse_track = np.zeros(n_samples, dtype=np.float32)
    # Gunfire peak amplitude: +6 dB relative to speech peak -> Peak_impulse = Peak_speech * 10^(6/20) ~ 2.0 * Peak_speech
    target_impulse_peak = peak_speech * (10.0 ** (6.0 / 20.0))

    for imp_t in impulse_times:
        if imp_t < duration:
            imp_sig = synthesize_gunfire_impulse(duration=0.10, sr=sr)
            imp_sig = imp_sig * target_impulse_peak
            start_i = int(imp_t * sr)
            end_i = min(start_i + len(imp_sig), n_samples)
            sig_slice_len = end_i - start_i
            impulse_track[start_i:end_i] += imp_sig[:sig_slice_len]

    # Combine: Noisy = Speech + Continuous Noise + Impulses
    total_noise = total_continuous_noise + impulse_track
    noisy_mixture = clean_speech + total_noise

    # Ensure no floating point clipping in mixture
    max_mix = np.max(np.abs(noisy_mixture))
    if max_mix > 0.98:
        norm_factor = 0.95 / max_mix
        noisy_mixture = noisy_mixture * norm_factor
        clean_speech = clean_speech * norm_factor
        total_noise = total_noise * norm_factor
        impulse_track = impulse_track * norm_factor

    actual_snr = 10.0 * np.log10(calculate_power(clean_speech) / calculate_power(total_noise))

    metadata = {
        "target_snr_db": snr_db,
        "actual_snr_db": float(actual_snr),
        "duration_sec": float(duration),
        "speech_peak": float(np.max(np.abs(clean_speech))),
        "impulse_peak": float(np.max(np.abs(impulse_track))),
        "impulse_times": impulse_times,
    }

    return (
        noisy_mixture.astype(np.float32),
        clean_speech.astype(np.float32),
        total_noise.astype(np.float32),
        metadata,
    )
