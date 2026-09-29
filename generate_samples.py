"""
VIKRAM-X: Dataset and Demo Audio Generation Pipeline
Generates clean speech, helicopter rotor noise, gunfire shockwave impulses,
and calibrated battlefield mixtures:
1. demo_battlefield_extreme.wav          (Overall SNR = -5 dB, Main SIH Showcase)
2. demo_battlefield_extreme_minus10.wav  (Overall SNR = -10 dB, Extreme Stress Test)
3. demo_battlefield_moderate.wav         (Overall SNR = 0 dB, Moderate Test)
4. demo_helicopter_only.wav              (Mode 1 Preset: Continuous Noise Only)
5. demo_impulsive_only.wav               (Mode 2 Preset: Impulsive Noise Only)
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio import (
    load_audio,
    save_audio,
    calculate_power,
    synthesize_helicopter_noise,
    synthesize_engine_wind,
    synthesize_gunfire_impulse,
    mix_battlefield_scenario,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPEECH_DIR = os.path.join(DATA_DIR, "speech")
HELI_DIR = os.path.join(DATA_DIR, "helicopter")
IMP_DIR = os.path.join(DATA_DIR, "impulse")
GEN_DIR = os.path.join(DATA_DIR, "generated")


def ensure_directories():
    for d in [SPEECH_DIR, HELI_DIR, IMP_DIR, GEN_DIR]:
        os.makedirs(d, exist_ok=True)


def acquire_clean_speech(sr: int = 16000) -> tuple[np.ndarray, str]:
    """Acquire standard clean human speech audio (14.84s LibriSpeech sample)."""
    clean_path = os.path.join(SPEECH_DIR, "clean_speech.wav")
    if os.path.exists(clean_path):
        y, _ = load_audio(clean_path, target_sr=sr)
        return y, clean_path

    raise FileNotFoundError(
        f"Clean speech reference not found at {clean_path}. Add a speech WAV there before generating scenarios."
    )


def generate_isolated_components(duration: float, sr: int = 16000):
    """Generate isolated continuous helicopter noise and gunshot impulse."""
    print("[2/5] Synthesizing acoustic components...")
    # Helicopter
    heli_path = os.path.join(HELI_DIR, "helicopter_noise.wav")
    heli_audio = synthesize_helicopter_noise(duration, sr=sr)
    save_audio(heli_path, heli_audio, sr=sr)
    print(f"      Helicopter noise saved to {heli_path}")

    # Gunfire
    imp_path = os.path.join(IMP_DIR, "gunfire_impulse.wav")
    imp_audio = synthesize_gunfire_impulse(duration=0.15, sr=sr)
    save_audio(imp_path, imp_audio, sr=sr)
    print(f"      Gunfire impulse saved to {imp_path}")

    return heli_audio, imp_audio


def generate_all_demo_scenarios():
    ensure_directories()
    sr = 16000

    # 1. Clean speech
    speech, speech_path = acquire_clean_speech(sr=sr)
    duration = len(speech) / sr

    # 2. Components
    generate_isolated_components(duration, sr=sr)

    print("[3/5] Generating Calibrated Battlefield Scenarios...")

    # Main Showcase: -5 dB SNR with impulses at 5.0s and 8.0s
    print("      - Creating demo_battlefield_extreme.wav (Target SNR = -5 dB)...")
    mix_m5, clean_m5, noise_m5, meta_m5 = mix_battlefield_scenario(
        clean_speech=speech,
        snr_db=-5.0,
        include_engine=True,
        impulse_times=[5.0, 8.0],
        sr=sr,
    )
    p_m5 = os.path.join(GEN_DIR, "demo_battlefield_extreme.wav")
    save_audio(p_m5, mix_m5, sr=sr)
    print(f"        Saved to {p_m5} (Actual SNR: {meta_m5['actual_snr_db']:.2f} dB)")

    # Extreme Stress Test: -10 dB SNR
    print("      - Creating demo_battlefield_extreme_minus10.wav (Target SNR = -10 dB)...")
    mix_m10, _, _, meta_m10 = mix_battlefield_scenario(
        clean_speech=speech,
        snr_db=-10.0,
        include_engine=True,
        impulse_times=[5.0, 8.0],
        sr=sr,
    )
    p_m10 = os.path.join(GEN_DIR, "demo_battlefield_extreme_minus10.wav")
    save_audio(p_m10, mix_m10, sr=sr)
    print(f"        Saved to {p_m10} (Actual SNR: {meta_m10['actual_snr_db']:.2f} dB)")

    # Moderate Test: 0 dB SNR
    print("      - Creating demo_battlefield_moderate.wav (Target SNR = 0 dB)...")
    mix_0, _, _, meta_0 = mix_battlefield_scenario(
        clean_speech=speech,
        snr_db=0.0,
        include_engine=True,
        impulse_times=[5.0, 8.0],
        sr=sr,
    )
    p_0 = os.path.join(GEN_DIR, "demo_battlefield_moderate.wav")
    save_audio(p_0, mix_0, sr=sr)
    print(f"        Saved to {p_0} (Actual SNR: {meta_0['actual_snr_db']:.2f} dB)")

    # Mode 1 Preset: Helicopter Only (Continuous Noise Only)
    print("      - Creating demo_helicopter_only.wav (Mode 1 Preset)...")
    mix_heli, _, _, meta_heli = mix_battlefield_scenario(
        clean_speech=speech,
        snr_db=-3.0,
        include_engine=False,
        impulse_times=[],
        sr=sr,
    )
    p_heli = os.path.join(GEN_DIR, "demo_helicopter_only.wav")
    save_audio(p_heli, mix_heli, sr=sr)
    print(f"        Saved to {p_heli}")

    # Mode 2 Preset: Impulsive Gunfire Only (No continuous helicopter)
    print("      - Creating demo_impulsive_only.wav (Mode 2 Preset)...")
    # Clean speech + impulses only (at 3.0s, 6.5s, 10.0s)
    p_speech = calculate_power(speech)
    peak_speech = float(np.max(np.abs(speech)))
    imp_track = np.zeros(len(speech), dtype=np.float32)
    target_imp_peak = peak_speech * (10.0 ** (6.0 / 20.0))  # +6 dB peak
    for t_imp in [3.0, 6.5, 10.0]:
        imp_s = synthesize_gunfire_impulse(duration=0.12, sr=sr) * target_imp_peak
        idx = int(t_imp * sr)
        end_idx = min(idx + len(imp_s), len(speech))
        imp_track[idx:end_idx] += imp_s[:end_idx - idx]

    mix_imp = speech + imp_track
    mix_imp = mix_imp / np.max(np.abs(mix_imp)) * 0.95
    p_imp = os.path.join(GEN_DIR, "demo_impulsive_only.wav")
    save_audio(p_imp, mix_imp, sr=sr)
    print(f"        Saved to {p_imp}")

    print("\n[SUCCESS] All calibrated battlefield audio files generated successfully!")


if __name__ == "__main__":
    generate_all_demo_scenarios()
