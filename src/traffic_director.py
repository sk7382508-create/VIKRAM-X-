"""
VIKRAM-X: Traffic Director & Intelligent Routing Decision Module
Frame-level deterministic classifier that routes frames to:
- CONTINUOUS NOISE -> AI ENHANCEMENT PATH
- IMPULSIVE EVENT  -> FAST DSP LIMITER PATH
- SPEECH DOMINANT  -> AI ENHANCEMENT / PRESERVATION PATH
"""

import numpy as np


STATE_IMPULSIVE = "IMPULSIVE_EVENT"
STATE_CONTINUOUS = "CONTINUOUS_NOISE"
STATE_SPEECH = "SPEECH_DOMINANT"

PATH_AI = "AI ENHANCEMENT"
PATH_DSP = "FAST DSP"
PATH_BYPASS = "PRESERVE / LIGHT AI"


class TrafficDirector:
    """
    Deterministic frame-by-frame traffic director.
    Evaluates temporal and spectral feature vectors to select optimal processing paths.
    """

    def __init__(
        self,
        kurtosis_threshold: float = 5.0,
        crest_factor_threshold_db: float = 13.8,
        flux_relative_threshold: float = 2.0,
        impulse_hold_frames: int = 6,  # Hold for ~60 ms after onset to catch reverberant transient tail
    ):
        self.kurt_thresh = kurtosis_threshold
        self.cf_thresh_db = crest_factor_threshold_db
        self.flux_thresh = flux_relative_threshold
        self.impulse_hold_frames = impulse_hold_frames

    def classify_frames(self, features: dict) -> dict:
        """
        Classify all frames and generate routing decisions.
        
        Returns:
        - conditions: list of state strings per frame
        - paths: list of chosen processing paths per frame
        - impulse_mask: boolean array (True where impulsive event active)
        - target_lambdas: target fusion weights (0.15 for DSP, 0.85 for AI)
        """
        n_frames = features["n_frames"]
        rms = features["rms"]
        crest_db = features["crest_factor_db"]
        kurtosis = features["kurtosis"]
        flux = features["spectral_flux"]
        low_band = features["low_band_energy"]
        mid_band = features["mid_band_energy"]
        est_snr = features["estimated_snr"]

        conditions = [STATE_CONTINUOUS] * n_frames
        paths = [PATH_AI] * n_frames
        impulse_mask = np.zeros(n_frames, dtype=bool)
        target_lambdas = np.full(n_frames, 0.85, dtype=np.float32)

        mean_rms = max(0.01, float(np.mean(rms)))
        median_flux = max(1e-4, float(np.median(flux)))

        # Pass 1: Detect instantaneous impulsive onsets
        onset_frames = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            # Criteria for explosive gunshot / ballistic shockwave:
            # High energy relative to signal, sharp crest factor spike, and high kurtosis or onset flux jump
            is_energy_spike = rms[i] >= (1.15 * mean_rms)
            is_crest_spike = crest_db[i] >= self.cf_thresh_db
            is_kurt_spike = kurtosis[i] >= self.kurt_thresh
            is_flux_spike = flux[i] >= (self.flux_thresh * median_flux)

            if is_energy_spike and is_crest_spike and (is_kurt_spike or is_flux_spike):
                onset_frames[i] = True

        # Pass 2: Apply hold-time envelope for impulsive events (transient protection window)
        hold_counter = 0
        for i in range(n_frames):
            if onset_frames[i]:
                hold_counter = self.impulse_hold_frames

            if hold_counter > 0:
                impulse_mask[i] = True
                conditions[i] = STATE_IMPULSIVE
                paths[i] = PATH_DSP
                target_lambdas[i] = 0.15  # DSP path strongly dominates
                hold_counter -= 1
            else:
                # Differentiate continuous noise vs clear speech
                # If high low-frequency energy (helicopter rotor) -> CONTINUOUS NOISE
                if est_snr[i] < 6.0 or (low_band[i] > 1.2 * mid_band[i]):
                    conditions[i] = STATE_CONTINUOUS
                    paths[i] = PATH_AI
                    target_lambdas[i] = 0.85  # AI continuous denoiser dominates
                else:
                    conditions[i] = STATE_SPEECH
                    paths[i] = PATH_AI
                    target_lambdas[i] = 0.75

        return {
            "conditions": conditions,
            "paths": paths,
            "impulse_mask": impulse_mask,
            "target_lambdas": target_lambdas,
        }

    @staticmethod
    def generate_timeline_summary(times: np.ndarray, conditions: list[str], paths: list[str]) -> list[dict]:
        """
        Merge adjacent frame classifications into clear, human-readable time blocks
        showing the exact architectural switching behavior over the audio duration.
        """
        if len(times) == 0:
            return []

        timeline = []
        current_state = conditions[0]
        current_path = paths[0]
        start_t = times[0]

        for i in range(1, len(times)):
            if conditions[i] != current_state:
                end_t = times[i - 1]
                timeline.append({
                    "start_time": float(round(start_t, 2)),
                    "end_time": float(round(end_t, 2)),
                    "condition": current_state,
                    "path": current_path,
                    "label": f"{start_t:.1f}s - {end_t:.1f}s: {current_state.replace('_', ' ')} -> {current_path}",
                })
                current_state = conditions[i]
                current_path = paths[i]
                start_t = times[i]

        # Final segment
        timeline.append({
            "start_time": float(round(start_t, 2)),
            "end_time": float(round(times[-1], 2)),
            "condition": current_state,
            "path": current_path,
            "label": f"{start_t:.1f}s - {times[-1]:.1f}s: {current_state.replace('_', ' ')} -> {current_path}",
        })

        return timeline
