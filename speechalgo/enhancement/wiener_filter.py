"""
Wiener Filter for speech enhancement.

The Wiener filter is an optimal linear filter in the minimum mean square
error (MMSE) sense. It provides theoretically optimal noise reduction by
estimating the clean speech spectrum based on signal and noise statistics.

Mathematical Foundation
-----------------------
Given noisy speech: Y(ω) = X(ω) + N(ω)

The Wiener filter gain is:
    G(ω) = P_x(ω) / (P_x(ω) + P_n(ω)) = P_x(ω) / P_y(ω)

where:
- P_x(ω) is the clean speech power spectrum
- P_n(ω) is the noise power spectrum
- P_y(ω) = P_x(ω) + P_n(ω) is the noisy speech power spectrum

The enhanced speech is:
    X̂(ω) = G(ω) * Y(ω)

Since P_x(ω) is unknown, we estimate it:
    P_x(ω) ≈ P_y(ω) - P_n(ω)

The gain can be rewritten as:
    G(ω) = max(0, 1 - P_n(ω)/P_y(ω))

This is equivalent to:
    G(ω) = max(0, 1 - 1/SNR(ω))

where SNR(ω) = P_x(ω) / P_n(ω) is the signal-to-noise ratio.

Properties:
- G(ω) → 1 when SNR is high (minimal attenuation)
- G(ω) → 0 when SNR is low (maximum attenuation)
- Optimal in MMSE sense for Gaussian signals

References
----------
.. [1] Wiener, N. (1949). "Extrapolation, interpolation, and smoothing of
       stationary time series". MIT Press.
.. [2] Lim, J. S., & Oppenheim, A. V. (1979). "Enhancement and bandwidth
       compression of noisy speech". Proceedings of the IEEE, 67(12).
.. [3] Scalart, P., & Filho, J. V. (1996). "Speech enhancement based on a
       priori signal to noise estimation". ICASSP 1996.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.utils.signal_processing import istft, stft
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class WienerFilter:
    """
    Apply Wiener filtering for optimal noise reduction.

    The Wiener filter computes an optimal gain function based on the
    signal-to-noise ratio to minimize mean square error.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    noise_floor : float, default=0.01
        Minimum gain floor to prevent over-suppression
    smoothing_alpha : float, default=0.9
        Temporal smoothing of gain (0.0-1.0)

    Attributes
    ----------
    noise_spectrum : ndarray or None
        Estimated noise power spectrum

    Examples
    --------
    >>> # Basic Wiener filtering
    >>> wiener = WienerFilter(sample_rate=16000)
    >>> # Estimate noise from initial silence
    >>> wiener.estimate_noise(noise_segment)
    >>> # Apply filtering
    >>> clean_audio = wiener.process(noisy_audio)

    >>> # Compare with spectral subtraction
    >>> from speechalgo.enhancement import SpectralSubtraction
    >>> spec_sub = SpectralSubtraction(sample_rate=16000)
    >>> spec_sub.estimate_noise(noise_segment)
    >>> clean_ss = spec_sub.process(noisy_audio)
    >>> clean_wiener = wiener.process(noisy_audio)

    >>> # Adjust noise floor for different scenarios
    >>> # Conservative (less artifacts): higher floor
    >>> wiener_conservative = WienerFilter(
    ...     sample_rate=16000,
    ...     noise_floor=0.1
    ... )
    >>> # Aggressive (more reduction): lower floor
    >>> wiener_aggressive = WienerFilter(
    ...     sample_rate=16000,
    ...     noise_floor=0.001
    ... )

    Notes
    -----
    Advantages:
    - Theoretically optimal (MMSE)
    - Less musical noise than spectral subtraction
    - Smooth gain function
    - Works well with stationary noise

    Limitations:
    - Requires noise spectrum estimation
    - Assumes Gaussian statistics
    - Performance degrades with non-stationary noise
    - May cause speech distortion at low SNR

    Comparison with Spectral Subtraction:
    - Wiener: Smoother, less artifacts, moderate reduction
    - Spectral Subtraction: More aggressive, more artifacts

    The noise floor parameter prevents complete signal suppression
    in high-noise regions, maintaining some speech intelligibility
    and reducing artifacts.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
        noise_floor: float = 0.01,
        smoothing_alpha: float = 0.9,
    ):
        validate_sample_rate(sample_rate)

        if not 0 <= noise_floor <= 1:
            raise ValueError(f"noise_floor must be in [0, 1], got {noise_floor}")

        if not 0 <= smoothing_alpha <= 1:
            raise ValueError(f"smoothing_alpha must be in [0, 1], got {smoothing_alpha}")

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_floor = noise_floor
        self.smoothing_alpha = smoothing_alpha

        self.noise_spectrum: Optional[npt.NDArray[np.float32]] = None

    def estimate_noise(self, noise_audio: npt.NDArray[np.float32]) -> None:
        """
        Estimate noise power spectrum from noise-only audio segment.

        Parameters
        ----------
        noise_audio : ndarray, shape (n_samples,)
            Audio segment containing only noise

        Examples
        --------
        >>> wiener = WienerFilter(sample_rate=16000)
        >>> noise_segment = noisy_audio[:8000]  # First 0.5s
        >>> wiener.estimate_noise(noise_segment)
        """
        validate_audio(noise_audio, min_length=self.frame_length)

        # Compute STFT of noise
        noise_stft = stft(
            noise_audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Compute average power spectrum
        noise_power = np.abs(noise_stft) ** 2
        self.noise_spectrum = np.mean(noise_power, axis=1)

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply Wiener filtering for noise reduction.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Noisy input audio signal

        Returns
        -------
        enhanced : ndarray
            Enhanced audio with reduced noise

        Raises
        ------
        ValueError
            If noise spectrum has not been estimated

        Examples
        --------
        >>> wiener = WienerFilter(sample_rate=16000)
        >>> wiener.estimate_noise(noise_segment)
        >>> clean = wiener.process(noisy_audio)
        """
        validate_audio(audio)

        if self.noise_spectrum is None:
            raise ValueError("Noise spectrum not estimated. Call estimate_noise() first.")

        # Compute STFT of noisy signal
        noisy_stft = stft(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Compute Wiener gain
        gain = self._compute_wiener_gain(noisy_stft)

        # Apply gain
        enhanced_stft = gain * noisy_stft

        # Inverse STFT
        enhanced = istft(enhanced_stft, hop_length=self.hop_length, window="hann")

        # Match length to input
        if len(enhanced) > len(audio):
            enhanced = enhanced[: len(audio)]
        elif len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))

        return enhanced.astype(np.float32)

    def _compute_wiener_gain(
        self, noisy_stft: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float32]:
        """
        Compute Wiener filter gain function.

        G(ω) = max(noise_floor, 1 - P_n(ω)/P_y(ω))

        Parameters
        ----------
        noisy_stft : ndarray, shape (n_freqs, n_frames)
            STFT of noisy signal

        Returns
        -------
        gain : ndarray, shape (n_freqs, n_frames)
            Wiener gain function
        """
        # Compute noisy power spectrum
        noisy_power = np.abs(noisy_stft) ** 2

        # Broadcast noise spectrum
        noise_power = self.noise_spectrum[:, np.newaxis]

        # Compute SNR-based gain: G = 1 - N/Y
        # Avoid division by zero
        gain = 1.0 - np.divide(
            noise_power,
            noisy_power,
            out=np.zeros_like(noisy_power),
            where=noisy_power > 1e-10,
        )

        # Apply noise floor
        gain = np.maximum(gain, self.noise_floor)

        # Ensure gain is in [0, 1]
        gain = np.clip(gain, 0.0, 1.0)

        # Apply temporal smoothing
        if self.smoothing_alpha > 0:
            gain = self._smooth_temporal(gain)

        return gain.astype(np.float32)

    def _smooth_temporal(self, gain: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply temporal smoothing to gain function.

        Parameters
        ----------
        gain : ndarray
            Gain function

        Returns
        -------
        smoothed : ndarray
            Smoothed gain function
        """
        smoothed = np.copy(gain)

        for t in range(1, gain.shape[1]):
            smoothed[:, t] = (
                self.smoothing_alpha * smoothed[:, t - 1] + (1 - self.smoothing_alpha) * gain[:, t]
            )

        return smoothed

    def __repr__(self) -> str:
        return f"WienerFilter(sample_rate={self.sample_rate}, " f"noise_floor={self.noise_floor})"
