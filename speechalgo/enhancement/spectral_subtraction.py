"""
Spectral Subtraction for noise reduction.

Spectral subtraction is one of the earliest and most widely studied speech
enhancement techniques. It operates in the frequency domain by estimating
and subtracting the noise spectrum from the noisy speech spectrum.

Mathematical Foundation
-----------------------
Given noisy speech: y(t) = x(t) + n(t)
where x(t) is clean speech and n(t) is additive noise.

In the frequency domain:
    Y(ω) = X(ω) + N(ω)

Basic spectral subtraction:
    |X̂(ω)| = |Y(ω)| - α|N̂(ω)|

where:
- |Y(ω)| is the noisy speech magnitude
- |N̂(ω)| is the estimated noise magnitude
- α is the oversubtraction factor (typically 1.0-2.0)
- |X̂(ω)| is the estimated clean speech magnitude

Power spectral subtraction:
    |X̂(ω)|² = |Y(ω)|² - α|N̂(ω)|²

To prevent negative values and musical noise:
1. Apply spectral floor: |X̂(ω)| = max(|X̂(ω)|, β|Y(ω)|)
2. Use oversubtraction: α > 1 for better noise reduction
3. Apply smoothing across time frames

The phase is typically preserved: ∠X̂(ω) = ∠Y(ω)

References
----------
.. [1] Boll, S. F. (1979). "Suppression of acoustic noise in speech using
       spectral subtraction". IEEE Transactions on Acoustics, Speech, and
       Signal Processing, 27(2), 113-120.
.. [2] Berouti, M., Schwartz, R., & Makhoul, J. (1979). "Enhancement of
       speech corrupted by acoustic noise". ICASSP 1979.
.. [3] Ephraim, Y., & Malah, D. (1984). "Speech enhancement using a
       minimum-mean square error short-time spectral amplitude estimator".
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.utils.signal_processing import istft, stft
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class SpectralSubtraction:
    """
    Reduce noise using spectral subtraction.

    This classic method estimates the noise spectrum and subtracts it
    from the noisy speech spectrum to recover the clean speech.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    oversubtraction_factor : float, default=1.5
        Oversubtraction parameter α (1.0-2.5). Higher values give more
        aggressive noise reduction but may introduce artifacts.
    spectral_floor : float, default=0.05
        Minimum gain β as fraction of noisy spectrum (0.0-0.2).
        Prevents over-suppression and reduces musical noise.
    smoothing_alpha : float, default=0.9
        Temporal smoothing factor (0.0-1.0). Higher values give smoother
        gain trajectories but less responsive to changes.

    Attributes
    ----------
    frame_length : int
        Frame length in samples
    hop_length : int
        Hop length in samples
    noise_spectrum : ndarray or None
        Estimated noise power spectrum

    Examples
    --------
    >>> # Basic usage with noise profile estimation
    >>> enhancer = SpectralSubtraction(sample_rate=16000)
    >>> # Estimate noise from initial silence (first 0.5s)
    >>> noise_segment = noisy_audio[:8000]
    >>> enhancer.estimate_noise(noise_segment)
    >>> # Enhance the full audio
    >>> clean_audio = enhancer.process(noisy_audio)

    >>> # Or provide pre-estimated noise spectrum
    >>> noise_spectrum = compute_noise_spectrum(noise_sample)
    >>> enhancer = SpectralSubtraction(sample_rate=16000)
    >>> enhancer.noise_spectrum = noise_spectrum
    >>> clean_audio = enhancer.process(noisy_audio)

    >>> # Adjust parameters for different noise levels
    >>> # High SNR (clean speech): lower oversubtraction
    >>> enhancer_light = SpectralSubtraction(
    ...     sample_rate=16000,
    ...     oversubtraction_factor=1.0,
    ...     spectral_floor=0.1
    ... )
    >>> # Low SNR (very noisy): higher oversubtraction
    >>> enhancer_aggressive = SpectralSubtraction(
    ...     sample_rate=16000,
    ...     oversubtraction_factor=2.0,
    ...     spectral_floor=0.02
    ... )

    Notes
    -----
    Advantages:
    - Simple and computationally efficient
    - Works well for stationary noise
    - No training required
    - Real-time capable

    Limitations:
    - Musical noise artifacts (random tones)
    - Requires noise-only segment for estimation
    - Poor performance with non-stationary noise
    - Speech distortion at low SNR

    Musical noise mitigation:
    - Increase spectral floor (β = 0.05-0.1)
    - Apply temporal smoothing (α = 0.8-0.95)
    - Use oversubtraction (α = 1.5-2.0)
    - Apply post-filtering

    Recommended settings:
    - Stationary noise (fan, hum): α=1.5, β=0.05
    - Babble noise: α=2.0, β=0.02
    - Clean environment: α=1.0, β=0.1
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
        oversubtraction_factor: float = 1.5,
        spectral_floor: float = 0.05,
        smoothing_alpha: float = 0.9,
    ):
        validate_sample_rate(sample_rate)

        if oversubtraction_factor < 0:
            raise ValueError(
                f"oversubtraction_factor must be non-negative, got {oversubtraction_factor}"
            )

        if not 0 <= spectral_floor <= 1:
            raise ValueError(f"spectral_floor must be in [0, 1], got {spectral_floor}")

        if not 0 <= smoothing_alpha <= 1:
            raise ValueError(f"smoothing_alpha must be in [0, 1], got {smoothing_alpha}")

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.oversubtraction_factor = oversubtraction_factor
        self.spectral_floor = spectral_floor
        self.smoothing_alpha = smoothing_alpha

        # Noise spectrum (to be estimated)
        self.noise_spectrum: Optional[npt.NDArray[np.float32]] = None

    def estimate_noise(self, noise_audio: npt.NDArray[np.float32]) -> None:
        """
        Estimate noise spectrum from noise-only audio segment.

        This should be called with a segment containing only noise
        (no speech), typically from the beginning or end of the recording.

        Parameters
        ----------
        noise_audio : ndarray, shape (n_samples,)
            Audio segment containing only noise

        Examples
        --------
        >>> enhancer = SpectralSubtraction(sample_rate=16000)
        >>> # Use first 0.5 seconds as noise estimate
        >>> noise_segment = noisy_audio[:8000]
        >>> enhancer.estimate_noise(noise_segment)
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
        Apply spectral subtraction to reduce noise.

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
        >>> enhancer = SpectralSubtraction(sample_rate=16000)
        >>> enhancer.estimate_noise(noise_segment)
        >>> clean = enhancer.process(noisy_audio)
        >>> print(f"Enhanced {len(clean)} samples")
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

        # Extract magnitude and phase
        noisy_magnitude = np.abs(noisy_stft)
        phase = np.angle(noisy_stft)

        # Apply spectral subtraction
        enhanced_magnitude = self._spectral_subtraction(noisy_magnitude)

        # Reconstruct complex spectrum
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)

        # Inverse STFT
        enhanced = istft(enhanced_stft, hop_length=self.hop_length, window="hann")

        # Match length to input
        if len(enhanced) > len(audio):
            enhanced = enhanced[: len(audio)]
        elif len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))

        return enhanced.astype(np.float32)

    def _spectral_subtraction(
        self, noisy_magnitude: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Apply spectral subtraction with smoothing and floor.

        Parameters
        ----------
        noisy_magnitude : ndarray, shape (n_freqs, n_frames)
            Magnitude spectrum of noisy signal

        Returns
        -------
        enhanced_magnitude : ndarray, shape (n_freqs, n_frames)
            Enhanced magnitude spectrum
        """
        n_freqs, n_frames = noisy_magnitude.shape

        # Broadcast noise spectrum to match shape
        noise_magnitude = np.sqrt(self.noise_spectrum[:, np.newaxis])

        # Power spectral subtraction
        noisy_power = noisy_magnitude**2
        noise_power = self.oversubtraction_factor * (noise_magnitude**2)

        # Subtract noise
        enhanced_power = noisy_power - noise_power

        # Apply spectral floor
        floor_power = (self.spectral_floor * noisy_magnitude) ** 2
        enhanced_power = np.maximum(enhanced_power, floor_power)

        # Convert back to magnitude
        enhanced_magnitude = np.sqrt(enhanced_power)

        # Apply temporal smoothing
        if self.smoothing_alpha > 0:
            enhanced_magnitude = self._smooth_temporal(enhanced_magnitude)

        return enhanced_magnitude

    def _smooth_temporal(self, magnitude: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply temporal smoothing to reduce musical noise.

        Uses first-order IIR filter: y[t] = α*y[t-1] + (1-α)*x[t]

        Parameters
        ----------
        magnitude : ndarray
            Magnitude spectrum

        Returns
        -------
        smoothed : ndarray
            Smoothed magnitude spectrum
        """
        smoothed = np.copy(magnitude)

        for t in range(1, magnitude.shape[1]):
            smoothed[:, t] = (
                self.smoothing_alpha * smoothed[:, t - 1]
                + (1 - self.smoothing_alpha) * magnitude[:, t]
            )

        return smoothed

    def __repr__(self) -> str:
        return (
            f"SpectralSubtraction(sample_rate={self.sample_rate}, "
            f"oversub={self.oversubtraction_factor}, "
            f"floor={self.spectral_floor})"
        )
