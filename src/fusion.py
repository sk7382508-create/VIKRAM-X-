"""
VIKRAM-X: Adaptive Fusion Controller Module
Fuses AI speech enhancement and Fast DSP impulsive suppression paths:
S_out(t) = lambda(t) * S_AI(t) + (1 - lambda(t)) * S_DSP(t)

Implements continuous sample-level IIR smoothing to ensure zero audible clicks
and transparently supports optional LMS residual cancellation.
"""

import numpy as np
import scipy.signal
from src.preprocessing import DEFAULT_SR, HOP_LENGTH


class FusionController:
    """
    VIKRAM-X Adaptive Fusion Controller.
    Dynamically balances AI neural enhancement and Fast DSP transient suppression.
    """

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr

    def fuse(
        self,
        ai_output: np.ndarray,
        dsp_output: np.ndarray,
        target_lambdas: np.ndarray,
        hop_length: int = HOP_LENGTH,
        smooth_time_ms: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adaptively fuse AI and DSP audio streams with smooth transitions.
        
        Args:
        - ai_output: enhanced audio from AI path
        - dsp_output: suppressed audio from Fast DSP path
        - target_lambdas: target lambda per frame (0.85 for continuous, 0.15 for impulse)
        - hop_length: STFT hop length
        - smooth_time_ms: smoothing time constant to avoid switching clicks
        
        Returns:
        - fused_audio: final combined communication audio
        - smoothed_lambda: sample-by-sample lambda(t) trajectory in [0, 1]
        """
        n_samples = min(len(ai_output), len(dsp_output))
        ai_sig = ai_output[:n_samples]
        dsp_sig = dsp_output[:n_samples]

        # 1. Expand frame-level lambdas to sample level
        n_frames = len(target_lambdas)
        frame_indices = np.arange(n_frames) * hop_length
        sample_indices = np.arange(n_samples)

        # Linear interpolation across frame centers
        raw_lambda = np.interp(
            sample_indices,
            frame_indices,
            target_lambdas,
            left=target_lambdas[0],
            right=target_lambdas[-1],
        )

        # 2. Continuous 1st-order IIR smoothing filter: lambda[n] = alpha * lambda[n-1] + (1 - alpha) * raw[n]
        # Fast transition ~20 ms
        tau_samples = (smooth_time_ms / 1000.0) * self.sr
        alpha = np.exp(-1.0 / max(1.0, tau_samples))

        smoothed_lambda = np.zeros(n_samples, dtype=np.float32)
        curr = float(raw_lambda[0])
        for i in range(n_samples):
            curr = alpha * curr + (1.0 - alpha) * float(raw_lambda[i])
            smoothed_lambda[i] = curr

        smoothed_lambda = np.clip(smoothed_lambda, 0.0, 1.0)

        # 3. Dynamic Fusion: S_out = lambda(t) * S_AI + (1 - lambda(t)) * S_DSP
        fused = smoothed_lambda * ai_sig + (1.0 - smoothed_lambda) * dsp_sig

        return fused.astype(np.float32), smoothed_lambda

    def apply_optional_lms(
        self,
        fused_signal: np.ndarray,
        reference_noise: np.ndarray | None = None,
        mu: float = 0.005,
        filter_order: int = 16,
    ) -> np.ndarray:
        """
        Optional Adaptive LMS Residual Cancellation.
        Flagged as 'Future enhancement' in UI if disabled.
        """
        if reference_noise is None or len(reference_noise) < len(fused_signal):
            return fused_signal

        n_samples = len(fused_signal)
        enhanced = np.zeros(n_samples, dtype=np.float32)
        weights = np.zeros(filter_order, dtype=np.float32)
        buffer = np.zeros(filter_order, dtype=np.float32)

        for n in range(n_samples):
            # Shift buffer
            buffer[1:] = buffer[:-1]
            buffer[0] = reference_noise[n]

            # Estimated residual noise
            y_est = np.dot(weights, buffer)
            err = fused_signal[n] - y_est
            enhanced[n] = err

            # Normalized LMS update
            norm = np.dot(buffer, buffer) + 1e-6
            weights += (2.0 * mu * err / norm) * buffer

        return enhanced
