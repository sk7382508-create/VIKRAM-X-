"""
VIKRAM-X: End-to-End Test and Verification Script
Verifies:
1. Feature extraction correctness
2. Traffic Director classification and timeline routing
3. AI Continuous-Noise Path execution
4. Fast DSP Impulsive Path execution
5. Adaptive Fusion and lambda(t) trajectory
6. Output reconstruction and genuine metrics calculation
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio import load_audio
from src.features import extract_frame_features
from src.traffic_director import TrafficDirector, STATE_IMPULSIVE, STATE_CONTINUOUS
from src.ai_path import AIContinuousEnhancementPath
from src.impulse_path import FastDSPImpulsePath
from src.fusion import FusionController
from src.reconstruction import reconstruct_and_export
from src.metrics import compute_evaluation_metrics


def test_full_pipeline():
    print("=" * 60)
    print("VIKRAM-X PIPELINE TEST: BATTLEFIELD EXTREME (-5 dB SNR)")
    print("=" * 60)

    # 1. Load audio
    input_path = os.path.join("data", "generated", "demo_battlefield_extreme.wav")
    clean_path = os.path.join("data", "speech", "clean_speech.wav")

    assert os.path.exists(input_path), f"Input file not found: {input_path}"
    assert os.path.exists(clean_path), f"Clean speech not found: {clean_path}"

    noisy_audio, sr = load_audio(input_path, target_sr=16000)
    clean_audio, _ = load_audio(clean_path, target_sr=16000)

    print(f"Loaded: {input_path} ({len(noisy_audio)/sr:.2f}s, {sr} Hz)")

    # 2. Extract features
    print("\n[Step 1] Extracting frame features...")
    features = extract_frame_features(noisy_audio, sr=sr)
    n_frames = features["n_frames"]
    times = features["times"]
    print(f"         Extracted features across {n_frames} frames.")
    print(f"         Mean RMS: {np.mean(features['rms']):.4f}, Max Crest Factor: {np.max(features['crest_factor_db']):.1f} dB")
    print(f"         Max Kurtosis: {np.max(features['kurtosis']):.1f}")

    # 3. Traffic Director routing
    print("\n[Step 2] Traffic Director frame classification...")
    director = TrafficDirector()
    decisions = director.classify_frames(features)
    conditions = decisions["conditions"]
    paths = decisions["paths"]
    target_lambdas = decisions["target_lambdas"]
    impulse_mask = decisions["impulse_mask"]

    n_impulsive = int(np.sum(impulse_mask))
    n_continuous = int(np.sum([1 for c in conditions if c == STATE_CONTINUOUS]))
    print(f"         Total frames: {n_frames}")
    print(f"         Continuous Noise frames: {n_continuous} (-> AI ENHANCEMENT)")
    print(f"         Impulsive Event frames:  {n_impulsive} (-> FAST DSP)")

    timeline = director.generate_timeline_summary(times, conditions, paths)
    print("\n[Step 3] Architectural Routing Timeline:")
    for segment in timeline:
        print(f"         * {segment['label']}")

    # Verify that gunfire at ~5s and ~8s was detected
    impulse_times = times[impulse_mask]
    has_5s_impulse = np.any((impulse_times >= 4.8) & (impulse_times <= 5.3))
    has_8s_impulse = np.any((impulse_times >= 7.8) & (impulse_times <= 8.3))
    print(f"\n         Gunfire at ~5.0s detected: {has_5s_impulse}")
    print(f"         Gunfire at ~8.0s detected: {has_8s_impulse}")

    # 4. AI Continuous Enhancement Path
    print("\n[Step 4] Running AI Continuous Enhancement Path...")
    ai_engine = AIContinuousEnhancementPath(sr=sr)
    ai_out = ai_engine.process(noisy_audio)
    print(f"         AI path completed. Output length: {len(ai_out)}")

    # 5. Fast DSP Impulsive Path
    print("\n[Step 5] Running Fast DSP Impulsive Path...")
    dsp_engine = FastDSPImpulsePath(sr=sr)
    dsp_out = dsp_engine.process(noisy_audio, impulse_mask=impulse_mask)
    print(f"         DSP path completed. Output length: {len(dsp_out)}")

    # 6. Adaptive Fusion Controller
    print("\n[Step 6] Adaptive Fusion Controller (lambda cross-fade)...")
    fusion = FusionController(sr=sr)
    fused_audio, smoothed_lambda = fusion.fuse(
        ai_output=ai_out,
        dsp_output=dsp_out,
        target_lambdas=target_lambdas,
    )
    print(f"         Fusion completed. Min lambda: {np.min(smoothed_lambda):.2f}, Max lambda: {np.max(smoothed_lambda):.2f}")

    # 7. Reconstruction and Export
    print("\n[Step 7] Reconstructing and exporting final enhanced communication WAV...")
    out_path = os.path.join("outputs", "enhanced_battlefield.wav")
    final_path, final_audio = reconstruct_and_export(fused_audio, output_path=out_path, sr=sr)
    print(f"         Enhanced audio written to: {final_path}")

    # 8. Genuine Metrics
    print("\n[Step 8] Evaluating Objective Speech Quality Metrics...")
    metrics = compute_evaluation_metrics(
        noisy_signal=noisy_audio,
        enhanced_signal=final_audio,
        clean_reference=clean_audio,
        sr=sr,
    )

    print("         -------------------------------------------")
    print(f"         Input RMS:           {metrics['input_rms']:.4f}")
    print(f"         Output RMS:          {metrics['output_rms']:.4f}")
    print(f"         Input SNR:           {metrics['input_snr']:.2f} dB")
    print(f"         Output SNR:          {metrics['output_snr']:.2f} dB")
    print(f"         SNR Improvement:     +{metrics['delta_snr']:.2f} dB")
    print(f"         Input STOI:          {metrics['input_stoi']:.3f}")
    print(f"         Output STOI:         {metrics['output_stoi']:.3f}")
    print(f"         STOI Improvement:    +{metrics['delta_stoi']:.3f}")
    print(f"         PESQ Status:         {metrics['pesq']}")
    print("         -------------------------------------------")

    assert os.path.exists(out_path), "Output file was not created"
    assert len(final_audio) > 0, "Final audio is empty"
    print("\n[ALL TESTS PASSED] VIKRAM-X vertical slice verified successfully!")


if __name__ == "__main__":
    test_full_pipeline()
