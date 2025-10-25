"""
Zero-Crossing Rate-based Voice Activity Detection.

The zero-crossing rate (ZCR) measures how often the signal crosses the
zero amplitude line. Voiced speech has low ZCR (periodic, low frequency),
unvoiced speech has high ZCR (noisy, high frequency), and silence typically
has moderate ZCR depending on background noise.

Mathematical Foundation
-----------------------
Zero-crossing rate for frame k:
    ZCR[k] = (1 / (2N)) * Σ_{n=1}^{N-1} |sign(x[n]) - sign(x[n-1])|

where sign(x) = 1 if x ≥ 0, else -1

ZCR characteristics:
- Silence: Low to moderate ZCR (depends on noise)
- Voiced speech: Low ZCR (fundamental frequency 80-300 Hz)
- Unvoiced speech: High ZCR (fricatives, stops)
- Noise: Variable ZCR

Combined decision (energy + ZCR):
    VAD[k] = 1 if (E[k] > E_th) AND (ZCR[k] < ZCR_th OR ZCR[k] in range)
             0 otherwise

References
----------
.. [1] Bachu, R. G., Kopparthi, S., Adapa, B., & Barkana, B. D. (2008).
       "Separation of voiced and unvoiced using zero crossing rate and energy
       of the speech signal". American Society for Engineering Education (ASEE).
.. [2] Rabiner, L. R., & Sambur, M. R. (1975). "An algorithm for determining
       the endpoints of isolated utterances". Bell System Technical Journal.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.preprocessing.framing import FrameExtractor
from speechalgo.utils.validation import (
    validate_audio,
    validate_frame_length,
    validate_sample_rate,
)
from speechalgo.vad.base import BaseVAD


class ZeroCrossingVAD(BaseVAD):
    """
    Detect voice activity using Zero-Crossing Rate combined with energy.

    This VAD method uses both ZCR and energy to detect speech. It works
    particularly well for distinguishing voiced speech, unvoiced speech,
    and silence.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    energy_threshold : float, optional
        Energy threshold. If None, uses adaptive threshold.
    zcr_threshold : float, optional
        ZCR threshold. If None, uses adaptive threshold.
    energy_ratio : float, default=0.1
        Ratio of mean energy for adaptive energy threshold
    zcr_ratio : float, default=1.5
        Ratio of mean ZCR for adaptive ZCR threshold
    min_speech_duration : float, default=0.1
        Minimum duration in seconds for a speech segment
    hangover : int, default=5
        Number of frames to extend speech activity

    Attributes
    ----------
    frame_extractor : FrameExtractor
        Frame extraction utility

    Examples
    --------
    >>> # Basic usage
    >>> vad = ZeroCrossingVAD(sample_rate=16000)
    >>> is_speech = vad.process(audio)

    >>> # Use fixed thresholds
    >>> vad = ZeroCrossingVAD(
    ...     sample_rate=16000,
    ...     energy_threshold=0.01,
    ...     zcr_threshold=0.15
    ... )
    >>> is_speech = vad.process(audio)

    >>> # Compare ZCR for different audio types
    >>> voiced = audio[is_voiced_segment]  # "aaa" sound
    >>> unvoiced = audio[is_unvoiced_segment]  # "sss" sound
    >>> print(f"Voiced ZCR: {compute_zcr(voiced):.3f}")
    >>> print(f"Unvoiced ZCR: {compute_zcr(unvoiced):.3f}")

    Notes
    -----
    ZCR-based VAD characteristics:
    - Effective for distinguishing voiced/unvoiced speech
    - Works well in moderate noise
    - Complementary to energy-based methods
    - Low computational cost

    Limitations:
    - Requires energy threshold tuning
    - May miss unvoiced speech if only using ZCR
    - Sensitive to DC offset (pre-processing needed)

    Typical ZCR values (normalized by sample rate):
    - Silence: 0.01 - 0.05
    - Voiced speech (vowels): 0.02 - 0.08
    - Unvoiced speech (fricatives): 0.1 - 0.3
    - High-frequency noise: 0.2 - 0.5
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
        energy_threshold: Optional[float] = None,
        zcr_threshold: Optional[float] = None,
        energy_ratio: float = 0.1,
        zcr_ratio: float = 1.5,
        min_speech_duration: float = 0.1,
        hangover: int = 5,
    ):
        super().__init__(sample_rate)

        validate_sample_rate(sample_rate)
        validate_frame_length(frame_length)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.energy_ratio = energy_ratio
        self.zcr_ratio = zcr_ratio
        self.min_speech_duration = min_speech_duration
        self.hangover = hangover

        # Create frame extractor
        self.frame_extractor = FrameExtractor(
            frame_length=frame_length,
            hop_length=hop_length,
            window=None,
        )

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """
        Detect voice activity using ZCR and energy.

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
        >>> vad = ZeroCrossingVAD(sample_rate=16000)
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> is_speech = vad.process(audio)
        >>> print(is_speech.shape)
        (63,)
        """
        validate_audio(audio)

        # Extract frames
        frames = self.frame_extractor.extract_frames(audio)

        # Compute energy and ZCR for each frame
        energy = self._compute_energy(frames)
        zcr = self._compute_zcr(frames)

        # Determine thresholds
        if self.energy_threshold is None:
            energy_th = self.energy_ratio * np.mean(energy)
        else:
            energy_th = self.energy_threshold

        if self.zcr_threshold is None:
            zcr_th = self.zcr_ratio * np.mean(zcr)
        else:
            zcr_th = self.zcr_threshold

        # Apply combined threshold
        # Speech: high energy AND (low ZCR for voiced OR moderate ZCR for unvoiced)
        vad_labels = (energy > energy_th) & (zcr < zcr_th)

        # Apply hangover
        vad_labels = self._apply_hangover(vad_labels)

        # Remove short segments
        vad_labels = self._remove_short_segments(vad_labels)

        return vad_labels

    def _compute_energy(self, frames: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute energy for each frame."""
        energy = np.sum(frames**2, axis=1)
        return energy

    def _compute_zcr(self, frames: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute zero-crossing rate for each frame.

        Parameters
        ----------
        frames : ndarray, shape (n_frames, frame_length)
            Audio frames

        Returns
        -------
        zcr : ndarray, shape (n_frames,)
            Zero-crossing rate for each frame
        """
        # Get signs of samples
        signs = np.sign(frames)
        signs[signs == 0] = 1  # Treat zero as positive

        # Count sign changes
        sign_changes = np.abs(np.diff(signs, axis=1))
        zcr = np.sum(sign_changes, axis=1) / (2.0 * self.frame_length)

        return zcr.astype(np.float32)

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
            f"ZeroCrossingVAD(sample_rate={self.sample_rate}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )
