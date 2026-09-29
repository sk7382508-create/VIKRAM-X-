"""
VIKRAM-X: Fast DSP Impulsive Noise Suppression Path
Ultra-fast deterministic DSP limiter and transient suppressor.
Applies sub-millisecond soft-knee transient clamping and dynamic envelope attenuation
to prevent deafening gunfire blasts from dominating the receiver, while preserving
the underlying speech intelligibility without muting.
"""

import numpy as np
from src.preprocessing import DEFAULT_SR, HOP_LENGTH, FRAME_LENGTH


class FastDSPImpulsePath:
    """
    Fast DSP path for impulsive transient suppression.
    Provides sub-millisecond dynamic range limiting and soft-saturation.
    """

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr

    def process(
        self,
        audio: np.ndarray,
        impulse_mask: np.ndarray | None = None,
        hop_length: int = HOP_LENGTH,
        max_attenuation_db: float = -18.0,
    ) -> np.ndarray:
        """
        Process audio with ultra-fast dynamic transient suppression.
        
        Args:
        - audio: input 1D float32 audio
        - impulse_mask: boolean array per frame indicating detected impulses
        - hop_length: hop size in samples for matching frame timestamps
        - max_attenuation_db: maximum blast attenuation applied during peak impulse
        """
        n_samples = len(audio)
        if n_samples == 0:
            return audio.copy()

        output = audio.copy()

        # 1. Establish running speech ceiling (3.0 * running speech RMS)
        # Using a 1-second rolling standard deviation to define normal speech amplitude range
        window_size = int(0.5 * self.sr)
        running_speech_ceiling = float(np.percentile(np.abs(audio), 90)) * 2.2
        speech_ceiling = max(0.08, min(0.35, running_speech_ceiling))

        # 2. Envelope follower with instantaneous attack (0.2 ms) and smooth release (25 ms)
        attack_alpha = np.exp(-1.0 / (0.0002 * self.sr))
        release_alpha = np.exp(-1.0 / (0.025 * self.sr))

        env = np.zeros(n_samples, dtype=np.float32)
        curr_env = 0.0
        abs_audio = np.abs(audio)

        for i in range(n_samples):
            val = abs_audio[i]
            if val > curr_env:
                curr_env = (1.0 - attack_alpha) * val + attack_alpha * curr_env
            else:
                curr_env = (1.0 - release_alpha) * val + release_alpha * curr_env
            env[i] = curr_env

        # 3. Soft-knee dynamic limiter & transient squashing
        # Compute gain reduction when envelope exceeds speech ceiling
        gain = np.ones(n_samples, dtype=np.float32)
        excess = env / (speech_ceiling + 1e-6)
        
        idx_excess = excess > 1.0
        # Smooth compression ratio for the transient
        gain[idx_excess] = (1.0 + np.log(excess[idx_excess])) / excess[idx_excess]

        # Apply additional targeted attenuation during frames identified by Traffic Director
        if impulse_mask is not None:
            n_frames = len(impulse_mask)
            frame_attenuation = np.ones(n_samples, dtype=np.float32)
            linear_atten = 10.0 ** (max_attenuation_db / 20.0)

            for f_idx in range(n_frames):
                if impulse_mask[f_idx]:
                    start_s = f_idx * hop_length
                    end_s = min(start_s + FRAME_LENGTH, n_samples)
                    frame_attenuation[start_s:end_s] = np.minimum(
                        frame_attenuation[start_s:end_s],
                        linear_atten,
                    )

            # Smooth frame attenuation envelope to avoid abrupt step changes
            b_smooth = np.hanning(int(0.015 * self.sr))  # 15 ms smoothing window
            b_smooth /= np.sum(b_smooth)
            frame_attenuation = np.convolve(frame_attenuation, b_smooth, mode="same")
            gain = gain * frame_attenuation

        # 4. Soft saturation curve to preserve speech harmonics without digital clipping
        attenuated = audio * gain
        # Tanh soft clipping ensures output never exceeds speech ceiling
        enhanced_dsp = speech_ceiling * np.tanh(attenuated / (speech_ceiling + 1e-6))

        return enhanced_dsp.astype(np.float32)
