"""
Spectral Entropy-based Voice Activity Detection.

Spectral entropy measures the randomness or unpredictability of the frequency
spectrum. Speech typically has lower entropy (more structure) compared to noise
which has higher entropy (more random). This makes spectral entropy a robust
feature for VAD, especially in noisy environments.

Mathematical Foundation
-----------------------
For a power spectrum P[k] with N bins, the normalized spectrum is:
    p[k] = P[k] / Σ_i P[i]

Spectral entropy:
    H = -Σ_{k=0}^{N-1} p[k] * log(p[k])

Normalized entropy:
    H_norm = H / log(N)

where H_norm ∈ [0, 1]:
- H_norm ≈ 0: Highly structured (few dominant frequencies) - likely speech
- H_norm ≈ 1: Uniform spectrum (white noise) - likely noise

Voice activity decision:
    VAD[k] = 1 if H_norm[k] < threshold
             0 otherwise

References
----------
.. [1] Shen, J. L., Hung, J. W., & Lee, L. S. (1998). "Robust entropy-based
       endpoint detection for speech recognition in noisy environments".
       ICSLP 1998.
.. [2] Renevey, P., & Drygajlo, A. (2001). "Entropy based voice activity
       detection in very noisy conditions". Eurospeech 2001.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.utils.signal_processing import stft
from speechalgo.utils.validation import (
    validate_audio,
    validate_frame_length,
    validate_sample_rate,
)
from speechalgo.vad.base import BaseVAD


class SpectralEntropyVAD(BaseVAD):
    """
    Detect voice activity using spectral entropy.

    This VAD method computes the entropy of the power spectrum for each
    frame. Speech has lower entropy (more structure) while noise has
    higher entropy (more random), enabling robust voice detection.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    threshold : float, optional
        Entropy threshold for voice detection. If None, uses adaptive threshold.
    threshold_ratio : float, default=0.7
        Ratio for adaptive threshold (used if threshold=None)
    min_speech_duration : float, default=0.1
        Minimum duration in seconds for a speech segment
    hangover : int, default=5
        Number of frames to extend speech activity

    Examples
    --------
    >>> # Basic usage with adaptive threshold
    >>> vad = SpectralEntropyVAD(sample_rate=16000)
    >>> is_speech = vad.process(audio)

    >>> # Use fixed threshold
    >>> vad = SpectralEntropyVAD(sample_rate=16000, threshold=0.6)
    >>> is_speech = vad.process(audio)

    >>> # Compare with energy-based VAD
    >>> from speechalgo.vad import EnergyBasedVAD
    >>> energy_vad = EnergyBasedVAD(sample_rate=16000)
    >>> entropy_vad = SpectralEntropyVAD(sample_rate=16000)
    >>> energy_result = energy_vad.process(noisy_audio)
    >>> entropy_result = entropy_vad.process(noisy_audio)
    >>> agreement = np.mean(energy_result == entropy_result)
    >>> print(f"Agreement: {agreement:.1%}")

    Notes
    -----
    Spectral entropy VAD advantages:
    - Robust to background noise
    - Works well with low SNR (<5dB)
    - Captures spectral structure differences
    - Less sensitive to level variations than energy-based VAD

    Disadvantages:
    - Higher computational cost (requires FFT)
    - May misclassify tonal noise as speech
    - Requires careful threshold tuning

    Best used when:
    - Operating in noisy environments
    - SNR is low or variable
    - Combined with energy-based features for robustness
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
        threshold: Optional[float] = None,
        threshold_ratio: float = 0.7,
        min_speech_duration: float = 0.1,
        hangover: int = 5,
    ):
        super().__init__(sample_rate)

        validate_sample_rate(sample_rate)
        validate_frame_length(frame_length)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold = threshold
        self.threshold_ratio = threshold_ratio
        self.min_speech_duration = min_speech_duration
        self.hangover = hangover

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """
        Detect voice activity using spectral entropy.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        vad_labels : ndarray, shape (n_frames,)
            Boolean array where True indicates voice activity

        Examples
        --------
        >>> vad = SpectralEntropyVAD(sample_rate=16000)
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> is_speech = vad.process(audio)
        >>> print(is_speech.shape)
        (63,)
        """
        validate_audio(audio)

        # Compute STFT
        stft_matrix = stft(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Compute power spectrum
        power_spectrum = np.abs(stft_matrix) ** 2

        # Compute entropy for each frame
        entropy = self._compute_entropy(power_spectrum)

        # Determine threshold
        if self.threshold is None:
            # For entropy, we want low values to indicate speech
            # So we use a threshold below the mean
            threshold = self.threshold_ratio * np.mean(entropy)
        else:
            threshold = self.threshold

        # Apply threshold (low entropy = speech)
        vad_labels = entropy < threshold

        # Apply hangover
        vad_labels = self._apply_hangover(vad_labels)

        # Remove short segments
        vad_labels = self._remove_short_segments(vad_labels)

        return vad_labels

    def _compute_entropy(self, power_spectrum: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute normalized spectral entropy for each frame.

        Parameters
        ----------
        power_spectrum : ndarray, shape (n_freqs, n_frames)
            Power spectrum

        Returns
        -------
        entropy : ndarray, shape (n_frames,)
            Normalized entropy values [0, 1] for each frame
        """
        # Add small constant to avoid log(0)
        power_spectrum = power_spectrum + 1e-10

        # Normalize to get probability distribution
        total_power = np.sum(power_spectrum, axis=0, keepdims=True)
        prob_dist = power_spectrum / total_power

        # Compute entropy: H = -Σ p*log(p)
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10), axis=0)

        # Normalize by log(N) to get [0, 1] range
        n_bins = power_spectrum.shape[0]
        normalized_entropy = entropy / np.log(n_bins)

        return normalized_entropy.astype(np.float32)

    def _apply_hangover(self, vad_labels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Apply hangover to extend speech activity."""
        if self.hangover == 0:
            return vad_labels

        extended_labels = np.copy(vad_labels)
        hangover_count = 0

        for i in range(len(vad_labels)):
            if vad_labels[i]:
                extended_labels[i] = True
                hangover_count = self.hangover
            elif hangover_count > 0:
                extended_labels[i] = True
                hangover_count -= 1

        return extended_labels

    def _remove_short_segments(self, vad_labels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """Remove speech segments shorter than minimum duration."""
        min_frames = int(self.min_speech_duration * self.sample_rate / self.hop_length)

        if min_frames <= 1:
            return vad_labels

        filtered_labels = np.copy(vad_labels)
        in_speech = False
        segment_start = 0

        for i in range(len(vad_labels)):
            if vad_labels[i] and not in_speech:
                in_speech = True
                segment_start = i
            elif not vad_labels[i] and in_speech:
                segment_length = i - segment_start
                if segment_length < min_frames:
                    filtered_labels[segment_start:i] = False
                in_speech = False

        if in_speech:
            segment_length = len(vad_labels) - segment_start
            if segment_length < min_frames:
                filtered_labels[segment_start:] = False

        return filtered_labels

    def __repr__(self) -> str:
        return (
            f"SpectralEntropyVAD(sample_rate={self.sample_rate}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )
