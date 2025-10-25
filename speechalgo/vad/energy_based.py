"""
Energy-based Voice Activity Detection.

Energy-based VAD is one of the simplest and most intuitive methods for
detecting speech. It computes the short-time energy of the signal and
compares it against a threshold to determine voice activity.

Mathematical Foundation
-----------------------
Short-time energy for frame k:
    E[k] = Σ_{n=0}^{N-1} (x[k*hop + n])^2

Voice activity decision:
    VAD[k] = 1 if E[k] > threshold
             0 otherwise

The threshold can be:
1. Fixed (absolute): threshold = α
2. Adaptive (relative): threshold = β * mean(E)
3. Dynamic: threshold adapts to background noise level

References
----------
.. [1] Sohn, J., Kim, N. S., & Sung, W. (1999). "A statistical model-based
       voice activity detection". IEEE Signal Processing Letters, 6(1), 1-3.
.. [2] Rabiner, L. R., & Sambur, M. R. (1975). "An algorithm for determining
       the endpoints of isolated utterances". Bell System Technical Journal, 54(2).
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


class EnergyBasedVAD(BaseVAD):
    """
    Detect voice activity using short-time energy.

    This VAD method computes the energy of each audio frame and compares
    it against a threshold. Frames above the threshold are classified as
    speech.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    threshold : float, optional
        Energy threshold for voice detection. If None, uses adaptive threshold.
    threshold_ratio : float, default=0.1
        Ratio of mean energy for adaptive threshold (used if threshold=None)
    min_speech_duration : float, default=0.1
        Minimum duration in seconds for a speech segment
    hangover : int, default=5
        Number of frames to extend speech activity after energy drops

    Attributes
    ----------
    frame_extractor : FrameExtractor
        Frame extraction utility

    Examples
    --------
    >>> # Basic usage with adaptive threshold
    >>> vad = EnergyBasedVAD(sample_rate=16000)
    >>> is_speech = vad.process(audio)
    >>> print(f"Speech frames: {is_speech.sum()} / {len(is_speech)}")
    Speech frames: 45 / 63

    >>> # Use fixed threshold
    >>> vad = EnergyBasedVAD(sample_rate=16000, threshold=0.01)
    >>> is_speech = vad.process(audio)

    >>> # Extract speech segments
    >>> speech_indices = np.where(is_speech)[0]
    >>> if len(speech_indices) > 0:
    ...     start_frame = speech_indices[0]
    ...     end_frame = speech_indices[-1]
    ...     start_sample = start_frame * vad.hop_length
    ...     end_sample = end_frame * vad.hop_length
    ...     speech = audio[start_sample:end_sample]

    Notes
    -----
    Energy-based VAD works well for:
    - Clean speech with moderate SNR (>10dB)
    - Controlled recording environments
    - Real-time applications (low computational cost)

    Limitations:
    - Sensitive to background noise
    - May fail with low SNR (<5dB)
    - Can miss unvoiced speech sounds (fricatives, stops)
    - Requires threshold tuning for different environments

    For better performance in noisy conditions, consider:
    - SpectralEntropyVAD
    - Combined features (energy + zero-crossing rate)
    - ML-based VAD (WebRTC VAD)
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
        threshold: Optional[float] = None,
        threshold_ratio: float = 0.1,
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

        # Create frame extractor without windowing for energy computation
        self.frame_extractor = FrameExtractor(
            frame_length=frame_length,
            hop_length=hop_length,
            window=None,
        )

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """
        Detect voice activity in audio signal.

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
        >>> vad = EnergyBasedVAD(sample_rate=16000)
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> is_speech = vad.process(audio)
        >>> print(is_speech.shape)
        (63,)
        """
        validate_audio(audio)

        # Extract frames
        frames = self.frame_extractor.extract_frames(audio)

        # Compute energy for each frame
        energy = self._compute_energy(frames)

        # Determine threshold
        if self.threshold is None:
            threshold = self.threshold_ratio * np.mean(energy)
        else:
            threshold = self.threshold

        # Apply threshold
        vad_labels = energy > threshold

        # Apply hangover (extend speech activity)
        vad_labels = self._apply_hangover(vad_labels)

        # Remove short speech segments
        vad_labels = self._remove_short_segments(vad_labels)

        return vad_labels

    def _compute_energy(self, frames: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute energy for each frame.

        Parameters
        ----------
        frames : ndarray, shape (n_frames, frame_length)
            Audio frames

        Returns
        -------
        energy : ndarray, shape (n_frames,)
            Energy values for each frame
        """
        energy = np.sum(frames**2, axis=1)
        return energy

    def _apply_hangover(self, vad_labels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """
        Extend speech activity by hangover frames.

        This helps maintain continuity in speech detection by extending
        detected speech regions for a few frames after energy drops.

        Parameters
        ----------
        vad_labels : ndarray
            Initial VAD labels

        Returns
        -------
        extended_labels : ndarray
            VAD labels with hangover applied
        """
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
        """
        Remove speech segments shorter than minimum duration.

        Parameters
        ----------
        vad_labels : ndarray
            VAD labels

        Returns
        -------
        filtered_labels : ndarray
            VAD labels with short segments removed
        """
        min_frames = int(self.min_speech_duration * self.sample_rate / self.hop_length)

        if min_frames <= 1:
            return vad_labels

        filtered_labels = np.copy(vad_labels)
        in_speech = False
        segment_start = 0

        for i in range(len(vad_labels)):
            if vad_labels[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                segment_start = i
            elif not vad_labels[i] and in_speech:
                # End of speech segment
                segment_length = i - segment_start
                if segment_length < min_frames:
                    # Remove short segment
                    filtered_labels[segment_start:i] = False
                in_speech = False

        # Check final segment
        if in_speech:
            segment_length = len(vad_labels) - segment_start
            if segment_length < min_frames:
                filtered_labels[segment_start:] = False

        return filtered_labels

    def __repr__(self) -> str:
        return (
            f"EnergyBasedVAD(sample_rate={self.sample_rate}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )
