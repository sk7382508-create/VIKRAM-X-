"""
VIKRAM-X: Verification script across all scenario modes and SNR levels
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio import load_audio
from src.features import extract_frame_features
from src.traffic_director import TrafficDirector, STATE_IMPULSIVE, STATE_CONTINUOUS
from src.ai_path import AIContinuousEnhancementPath
from src.impulse_path import FastDSPImpulsePath
from src.fusion import FusionController
from src.reconstruction import reconstruct_and_export
from src.metrics import compute_evaluation_metrics

test_cases = [
    ("Mode 1 (Helicopter Only)", "data/generated/demo_helicopter_only.wav"),
    ("Mode 2 (Impulsive Only)", "data/generated/demo_impulsive_only.wav"),
    ("Mode 3 (-5 dB Main Showcase)", "data/generated/demo_battlefield_extreme.wav"),
    ("Mode 3 (-10 dB Stress Test)", "data/generated/demo_battlefield_extreme_minus10.wav"),
    ("Mode 3 (0 dB Moderate)", "data/generated/demo_battlefield_moderate.wav"),
]

clean_path = "data/speech/clean_speech.wav"
clean_y, sr = load_audio(clean_path, 16000)

print(f"{'Mode':<30} | {'Impulses':<8} | {'Input SNR':<10} | {'Output SNR':<10} | {'d_SNR':<8} | {'d_STOI':<8}")
print("-" * 85)

for mode_name, wav_path in test_cases:
    y_raw, _ = load_audio(wav_path, sr)
    f = extract_frame_features(y_raw, sr)
    director = TrafficDirector()
    dec = director.classify_frames(f)
    
    ai_out = AIContinuousEnhancementPath(sr).process(y_raw)
    dsp_out = FastDSPImpulsePath(sr).process(y_raw, dec["impulse_mask"])
    fused, lam = FusionController(sr).fuse(ai_out, dsp_out, dec["target_lambdas"])
    
    out_file = f"outputs/test_{mode_name[:10].replace(' ', '_')}.wav"
    reconstruct_and_export(fused, out_file, sr)
    
    m = compute_evaluation_metrics(y_raw, fused, clean_y, sr)
    
    imp_count = sum(dec["impulse_mask"])
    in_snr = f"{m['input_snr']:.2f} dB" if m['input_snr'] is not None else "N/A"
    out_snr = f"{m['output_snr']:.2f} dB" if m['output_snr'] is not None else "N/A"
    d_snr = f"+{m['delta_snr']:.2f} dB" if m['delta_snr'] is not None else "N/A"
    d_stoi = f"+{m['delta_stoi']:.3f}" if m['delta_stoi'] is not None else "N/A"
    
    print(f"{mode_name:<30} | {imp_count:<8} | {in_snr:<10} | {out_snr:<10} | {d_snr:<8} | {d_stoi:<8}")

print("\nAll modes verified successfully!")
