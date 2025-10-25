"""
Temporal (time-domain) features for audio analysis.

Temporal features characterize the time-domain properties of audio signals.
They are computationally efficient and provide complementary information
to spectral features.

Mathematical Foundations
------------------------

1. Zero-Crossing Rate (ZCR):
   ZCR = (1/(2N)) * Σ_{n=1}^{N-1} |sign(x[n]) - sign(x[n-1])|

   Indicates the rate at which the signal changes sign.
   - High ZCR: Noisy, high-frequency content (fricatives, noise)
   - Low ZCR: Tonal, low-frequency content (vowels, music)

2. Short-Time Energy:
   E = Σ_{n=0}^{N-1} x[n]²

   Measures the signal power in each frame.
   - High energy: Loud, emphasized sounds
   - Low energy: Quiet, weak sounds

3. Root Mean Square (RMS):
   RMS = sqrt((1/N) * Σ_{n=0}^{N-1} x[n]²) = sqrt(E/N)

   Normalized energy measure, indicates amplitude.

References
----------
.. [1] Rabiner, L., & Juang, B. H. (1993). "Fundamentals of speech recognition".
.. [2] Bachu, R. G., et al. (2008). "Separation of voiced and unvoiced using
       zero crossing rate and energy of the speech signal".
.. [3] Kedem, B. (1986). "Spectral analysis and discrimination by zero-crossings".
"""

import numpy as np
import numpy.typing as npt

from speechalgo.preprocessing.framing import FrameExtractor
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class TemporalFeatures:
    """
    Extract temporal (time-domain) features from audio signals.

    This class provides methods to compute time-domain features that
    characterize the temporal properties of audio.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames

    Examples
    --------
    >>> # Initialize feature extractor
    >>> temporal = TemporalFeatures(sample_rate=16000)

    >>> # Compute temporal features
    >>> zcr = temporal.zero_crossing_rate(audio)
    >>> energy = temporal.short_time_energy(audio)
    >>> rms = temporal.root_mean_square(audio)

    >>> # Analyze speech characteristics
    >>> voiced_segment = audio[is_voiced]
    >>> unvoiced_segment = audio[is_unvoiced]
    >>> print(f"Voiced ZCR: {temporal.zero_crossing_rate(voiced_segment).mean():.3f}")
    >>> print(f"Unvoiced ZCR: {temporal.zero_crossing_rate(unvoiced_segment).mean():.3f}")

    >>> # Visualize features
    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(len(zcr)) * 256 / 16000
    >>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    >>> ax1.plot(time, zcr)
    >>> ax1.set_ylabel('ZCR')
    >>> ax2.plot(time, energy)
    >>> ax2.set_ylabel('Energy')
    >>> ax3.plot(time, rms)
    >>> ax3.set_ylabel('RMS')
    >>> ax3.set_xlabel('Time (s)')
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    Temporal features are useful for:
    - Voice activity detection
    - Voiced/unvoiced classification
    - Speech/music discrimination
    - Audio segmentation
    - Energy normalization

    Feature interpretations:
    - High ZCR: Fricatives ('s', 'f'), noise, percussion
    - Low ZCR: Vowels, pitched sounds, bass
    - High energy: Emphasized speech, loud sounds
    - Low energy: Whispers, background, silence
    - High RMS: Strong signal, good SNR
    - Low RMS: Weak signal, low SNR

    Advantages:
    - Very fast to compute (no FFT required)
    - Low memory requirements
    - Suitable for real-time processing
    - Intuitive interpretation

    Often combined with spectral features for robust analysis.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        validate_sample_rate(sample_rate)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Frame extractor (no windowing for temporal features)
        self.frame_extractor = FrameExtractor(
            frame_length=frame_length,
            hop_length=hop_length,
            window=None,
        )

    def zero_crossing_rate(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute zero-crossing rate.

        ZCR measures how often the signal crosses zero amplitude,
        indicating frequency content and periodicity.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        zcr : ndarray, shape (n_frames,)
            Zero-crossing rate for each frame [0, 1]

        Examples
        --------
        >>> temporal = TemporalFeatures(sample_rate=16000)
        >>> zcr = temporal.zero_crossing_rate(audio)
        >>> print(f"Mean ZCR: {zcr.mean():.3f}")

        >>> # Distinguish voiced/unvoiced
        >>> voiced_threshold = 0.1
        >>> is_voiced = zcr < voiced_threshold
        >>> print(f"Voiced frames: {is_voiced.sum()} / {len(is_voiced)}")
        """
        validate_audio(audio)

        # Extract frames
        frames = self.frame_extractor.extract_frames(audio)

        # Compute ZCR for each frame
        zcr = self._compute_zcr(frames)

        return zcr

    def short_time_energy(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute short-time energy.

        Energy measures the signal power in each frame, useful for
        detecting speech activity and emphasis.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        energy : ndarray, shape (n_frames,)
            Short-time energy for each frame

        Examples
        --------
        >>> temporal = TemporalFeatures(sample_rate=16000)
        >>> energy = temporal.short_time_energy(audio)
        >>> print(f"Energy range: {energy.min():.2e} - {energy.max():.2e}")

        >>> # Voice activity detection
        >>> threshold = 0.01 * energy.max()
        >>> is_speech = energy > threshold
        """
        validate_audio(audio)

        # Extract frames
        frames = self.frame_extractor.extract_frames(audio)

        # Compute energy for each frame
        energy = np.sum(frames**2, axis=1)

        return energy.astype(np.float32)

    def root_mean_square(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute root mean square (RMS) energy.

        RMS provides a normalized energy measure that indicates the
        signal amplitude level.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        rms : ndarray, shape (n_frames,)
            RMS energy for each frame

        Examples
        --------
        >>> temporal = TemporalFeatures(sample_rate=16000)
        >>> rms = temporal.root_mean_square(audio)
        >>> print(f"Mean RMS: {rms.mean():.4f}")

        >>> # Convert to dB
        >>> rms_db = 20 * np.log10(rms + 1e-10)
        >>> print(f"RMS range: {rms_db.min():.1f} - {rms_db.max():.1f} dB")
        """
        validate_audio(audio)

        # Extract frames
        frames = self.frame_extractor.extract_frames(audio)

        # Compute RMS for each frame
        rms = np.sqrt(np.mean(frames**2, axis=1))

        return rms.astype(np.float32)

    def _compute_zcr(self, frames: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute zero-crossing rate for frames.

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

    def __repr__(self) -> str:
        return (
            f"TemporalFeatures(sample_rate={self.sample_rate}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )
