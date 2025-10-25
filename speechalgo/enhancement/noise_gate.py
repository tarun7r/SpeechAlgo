"""
Noise Gate for simple threshold-based noise suppression.

A noise gate is a simple audio processor that attenuates signals below
a threshold. It's one of the most straightforward noise reduction methods,
effective for removing low-level background noise during silence periods.

Mathematical Foundation
-----------------------
Basic gate operation:
    y(t) = x(t) * G(t)

where the gain G(t) is:
    G(t) = 1    if level(t) > threshold
         = 0    if level(t) < threshold - range

The level can be measured using:
- Peak amplitude: level(t) = |x(t)|
- RMS energy: level(t) = sqrt(mean(x²)) over a window
- dB SPL: level_dB(t) = 20*log10(level(t))

Improvements:
1. Soft knee: Smooth transition around threshold
2. Attack time: How quickly gate opens
3. Release time: How quickly gate closes
4. Hold time: Minimum open duration
5. Range: Maximum attenuation (instead of complete cutoff)

Soft knee gain with attack/release:
    G_target(t) = smoothstep(level(t), threshold-range, threshold)
    G(t) = smooth(G_target(t), attack_time, release_time)

References
----------
.. [1] Zölzer, U. (2011). "DAFX: Digital audio effects" (2nd ed.). Wiley.
.. [2] Reiss, J. D., & McPherson, A. (2014). "Audio effects: Theory,
       implementation and application". CRC Press.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.preprocessing.framing import FrameExtractor
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class NoiseGate:
    """
    Simple noise gate for threshold-based noise suppression.

    A noise gate attenuates audio below a threshold, effectively removing
    low-level background noise during silence and quiet passages.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    threshold_db : float, default=-40.0
        Gate threshold in dB. Audio below this level is attenuated.
        Typical range: -60 to -20 dB
    range_db : float, default=60.0
        Maximum attenuation in dB when gate is closed.
        Use 60-80 for near-silence, lower values for gentler effect.
    attack_time : float, default=0.005
        Attack time in seconds (how quickly gate opens).
        Typical: 0.001-0.010 seconds
    release_time : float, default=0.05
        Release time in seconds (how quickly gate closes).
        Typical: 0.01-0.2 seconds
    hold_time : float, default=0.01
        Hold time in seconds (minimum time gate stays open).
        Typical: 0.005-0.05 seconds
    frame_length : int, default=512
        Frame length for level detection
    hop_length : int, default=256
        Hop length for processing

    Attributes
    ----------
    threshold_linear : float
        Linear threshold value
    range_linear : float
        Linear range value

    Examples
    --------
    >>> # Basic noise gate
    >>> gate = NoiseGate(sample_rate=16000, threshold_db=-40)
    >>> gated_audio = gate.process(noisy_audio)

    >>> # Aggressive gate for very noisy recordings
    >>> gate_aggressive = NoiseGate(
    ...     sample_rate=16000,
    ...     threshold_db=-30,
    ...     range_db=80,
    ...     release_time=0.02
    ... )
    >>> gated = gate_aggressive.process(very_noisy_audio)

    >>> # Gentle gate for mild noise
    >>> gate_gentle = NoiseGate(
    ...     sample_rate=16000,
    ...     threshold_db=-50,
    ...     range_db=20,
    ...     attack_time=0.001,
    ...     release_time=0.1
    ... )
    >>> gated = gate_gentle.process(mildly_noisy_audio)

    >>> # Compare before and after
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    >>> ax1.plot(noisy_audio)
    >>> ax1.set_title('Before Noise Gate')
    >>> ax2.plot(gated_audio)
    >>> ax2.set_title('After Noise Gate')
    >>> plt.show()

    Notes
    -----
    Advantages:
    - Very simple and fast
    - No latency (can be real-time)
    - Effective for removing silence/pause noise
    - No spectral artifacts
    - Transparent during speech

    Limitations:
    - Only reduces noise during silence
    - No reduction during speech
    - Can cut off quiet speech beginnings/endings
    - Not effective for continuous background noise

    Best used for:
    - Removing room noise during pauses
    - Cleaning up recordings with silence
    - Pre-processing before other enhancement
    - Real-time applications

    Not recommended for:
    - Continuous background noise (use spectral methods)
    - Preserving very quiet speech
    - Music with quiet passages

    Parameter tuning guidelines:
    - threshold_db: Set just above noise floor
    - range_db: 60 dB gives near-complete silence
    - attack_time: Fast (1-5ms) to avoid cutting speech onset
    - release_time: Slow (50-200ms) to avoid choppy sound
    - hold_time: Long enough to bridge short pauses
    """

    def __init__(
        self,
        sample_rate: int,
        threshold_db: float = -40.0,
        range_db: float = 60.0,
        attack_time: float = 0.005,
        release_time: float = 0.05,
        hold_time: float = 0.01,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        validate_sample_rate(sample_rate)

        if range_db < 0:
            raise ValueError(f"range_db must be non-negative, got {range_db}")

        if attack_time < 0:
            raise ValueError(f"attack_time must be non-negative, got {attack_time}")

        if release_time < 0:
            raise ValueError(f"release_time must be non-negative, got {release_time}")

        if hold_time < 0:
            raise ValueError(f"hold_time must be non-negative, got {hold_time}")

        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.range_db = range_db
        self.attack_time = attack_time
        self.release_time = release_time
        self.hold_time = hold_time
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Convert dB to linear
        self.threshold_linear = self._db_to_linear(threshold_db)
        self.range_linear = self._db_to_linear(-range_db)

        # Compute time constants in frames
        self.attack_frames = max(1, int(attack_time * sample_rate / hop_length))
        self.release_frames = max(1, int(release_time * sample_rate / hop_length))
        self.hold_frames = max(1, int(hold_time * sample_rate / hop_length))

        # Frame extractor
        self.frame_extractor = FrameExtractor(
            frame_length=frame_length, hop_length=hop_length, window=None
        )

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply noise gate to audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        gated : ndarray
            Audio with noise gate applied

        Examples
        --------
        >>> gate = NoiseGate(sample_rate=16000)
        >>> gated = gate.process(noisy_audio)
        >>> # Check noise reduction in silence
        >>> silence_power_before = np.mean(noisy_audio[0:1000]**2)
        >>> silence_power_after = np.mean(gated[0:1000]**2)
        >>> reduction_db = 10 * np.log10(silence_power_before / silence_power_after)
        >>> print(f"Noise reduced by {reduction_db:.1f} dB")
        """
        validate_audio(audio)

        # Extract frames for level detection
        frames = self.frame_extractor.extract_frames(audio)

        # Compute RMS level for each frame
        levels = self._compute_levels(frames)

        # Compute gain envelope
        gain_envelope = self._compute_gain_envelope(levels)

        # Interpolate gain to sample-by-sample
        gain_samples = self._interpolate_gain(gain_envelope, len(audio))

        # Apply gain
        gated = audio * gain_samples

        return gated.astype(np.float32)

    def _compute_levels(self, frames: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute RMS level for each frame.

        Parameters
        ----------
        frames : ndarray, shape (n_frames, frame_length)
            Audio frames

        Returns
        -------
        levels : ndarray, shape (n_frames,)
            RMS level for each frame
        """
        # Compute RMS: sqrt(mean(x^2))
        levels = np.sqrt(np.mean(frames**2, axis=1))
        return levels

    def _compute_gain_envelope(self, levels: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute gain envelope with attack, release, and hold.

        Parameters
        ----------
        levels : ndarray
            Frame-by-frame level measurements

        Returns
        -------
        gain : ndarray
            Gain envelope
        """
        n_frames = len(levels)
        gain = np.ones(n_frames, dtype=np.float32)
        hold_counter = 0

        for i in range(n_frames):
            # Determine target gain based on level
            if levels[i] > self.threshold_linear:
                # Above threshold: gate open
                target_gain = 1.0
                hold_counter = self.hold_frames
            elif hold_counter > 0:
                # In hold period: keep gate open
                target_gain = 1.0
                hold_counter -= 1
            else:
                # Below threshold and not holding: apply range
                target_gain = self.range_linear

            # Apply attack/release smoothing
            if i > 0:
                if target_gain > gain[i - 1]:
                    # Opening: use attack time
                    alpha = 1.0 / self.attack_frames
                else:
                    # Closing: use release time
                    alpha = 1.0 / self.release_frames

                gain[i] = gain[i - 1] + alpha * (target_gain - gain[i - 1])
            else:
                gain[i] = target_gain

        return gain

    def _interpolate_gain(
        self, gain_envelope: npt.NDArray[np.float32], n_samples: int
    ) -> npt.NDArray[np.float32]:
        """
        Interpolate frame-by-frame gain to sample-by-sample.

        Parameters
        ----------
        gain_envelope : ndarray, shape (n_frames,)
            Gain values per frame
        n_samples : int
            Number of samples in output

        Returns
        -------
        gain_samples : ndarray, shape (n_samples,)
            Interpolated gain values
        """
        # Create time points for frames
        frame_times = np.arange(len(gain_envelope)) * self.hop_length

        # Create time points for samples
        sample_times = np.arange(n_samples)

        # Linear interpolation
        gain_samples = np.interp(sample_times, frame_times, gain_envelope)

        return gain_samples.astype(np.float32)

    @staticmethod
    def _db_to_linear(db: float) -> float:
        """Convert dB to linear amplitude."""
        return 10.0 ** (db / 20.0)

    def __repr__(self) -> str:
        return (
            f"NoiseGate(sample_rate={self.sample_rate}, "
            f"threshold={self.threshold_db}dB, range={self.range_db}dB)"
        )
