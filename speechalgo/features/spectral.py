"""
Spectral features for audio analysis.

Spectral features capture the frequency content and distribution of
audio signals. They are widely used in speech and music information
retrieval, audio classification, and signal characterization.

Mathematical Foundations
------------------------

1. Spectral Centroid: Center of mass of the spectrum
   C = Σ(f[k] * |X[k]|) / Σ|X[k]|
   Indicates perceived "brightness" of sound.

2. Spectral Rolloff: Frequency below which X% of energy is contained
   R = frequency where Σ_{k=0}^{R} |X[k]|² ≥ α * Σ_{k=0}^{N} |X[k]|²
   Typically α = 0.85 or 0.95.

3. Spectral Flux: Change in spectral magnitude over time
   F[t] = Σ_k (|X[k,t]| - |X[k,t-1]|)²
   Indicates spectral variability or onset detection.

4. Spectral Bandwidth: Weighted standard deviation of frequencies
   BW = sqrt(Σ(f[k] - C)² * |X[k]| / Σ|X[k]|)
   Measures spread around the centroid.

References
----------
.. [1] Peeters, G. (2004). "A large set of audio features for sound description".
       CUIDADO Project.
.. [2] McFee, B., et al. (2015). "librosa: Audio and music signal analysis in python".
.. [3] Tzanetakis, G., & Cook, P. (2002). "Musical genre classification of audio
       signals". IEEE Transactions on Speech and Audio Processing.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.utils.signal_processing import stft
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class SpectralFeatures:
    """
    Extract spectral features from audio signals.

    This class provides methods to compute various frequency-domain
    features that characterize the spectral properties of audio.

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
    >>> spectral = SpectralFeatures(sample_rate=16000)

    >>> # Compute spectral centroid
    >>> centroid = spectral.spectral_centroid(audio)
    >>> print(f"Mean centroid: {np.mean(centroid):.1f} Hz")

    >>> # Compute multiple features
    >>> centroid = spectral.spectral_centroid(audio)
    >>> rolloff = spectral.spectral_rolloff(audio)
    >>> flux = spectral.spectral_flux(audio)
    >>> bandwidth = spectral.spectral_bandwidth(audio)

    >>> # Visualize features
    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(len(centroid)) * 256 / 16000
    >>> plt.figure(figsize=(12, 8))
    >>> plt.subplot(4, 1, 1)
    >>> plt.plot(time, centroid)
    >>> plt.ylabel('Centroid (Hz)')
    >>> plt.subplot(4, 1, 2)
    >>> plt.plot(time, rolloff)
    >>> plt.ylabel('Rolloff (Hz)')
    >>> plt.subplot(4, 1, 3)
    >>> plt.plot(time, flux)
    >>> plt.ylabel('Flux')
    >>> plt.subplot(4, 1, 4)
    >>> plt.plot(time, bandwidth)
    >>> plt.ylabel('Bandwidth (Hz)')
    >>> plt.xlabel('Time (s)')
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    Spectral features are useful for:
    - Music genre classification
    - Speech/music discrimination
    - Audio event detection
    - Instrument identification
    - Audio quality assessment

    Feature interpretations:
    - High centroid: Bright, sharp sounds (cymbals, consonants)
    - Low centroid: Dark, mellow sounds (bass, vowels)
    - High rolloff: Broadband signals
    - Low rolloff: Narrowband signals
    - High flux: Transients, onsets, changes
    - Low flux: Steady-state sounds
    - High bandwidth: Noise-like, spread spectrum
    - Low bandwidth: Tonal, narrow spectrum
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

    def spectral_centroid(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute spectral centroid (center of mass of spectrum).

        The spectral centroid indicates where the "center of mass" of
        the spectrum is located, correlating with perceived brightness.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        centroid : ndarray, shape (n_frames,)
            Spectral centroid in Hz for each frame

        Examples
        --------
        >>> spectral = SpectralFeatures(sample_rate=16000)
        >>> centroid = spectral.spectral_centroid(audio)
        >>> print(f"Centroid range: {centroid.min():.1f} - {centroid.max():.1f} Hz")
        """
        validate_audio(audio)

        # Compute magnitude spectrum
        magnitude = self._compute_magnitude_spectrum(audio)

        # Frequency bins in Hz
        freqs = np.fft.rfftfreq(self.frame_length, 1 / self.sample_rate)

        # Compute centroid: weighted mean frequency
        centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / (
            np.sum(magnitude, axis=0) + 1e-10
        )

        return centroid.astype(np.float32)

    def spectral_rolloff(
        self, audio: npt.NDArray[np.float32], percentile: float = 0.85
    ) -> npt.NDArray[np.float32]:
        """
        Compute spectral rolloff frequency.

        The rolloff frequency is the frequency below which a specified
        percentage of the total spectral energy is contained.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal
        percentile : float, default=0.85
            Percentile for rolloff calculation (0.0 to 1.0)

        Returns
        -------
        rolloff : ndarray, shape (n_frames,)
            Spectral rolloff frequency in Hz for each frame

        Examples
        --------
        >>> spectral = SpectralFeatures(sample_rate=16000)
        >>> rolloff_85 = spectral.spectral_rolloff(audio, percentile=0.85)
        >>> rolloff_95 = spectral.spectral_rolloff(audio, percentile=0.95)
        """
        validate_audio(audio)

        # Compute magnitude spectrum
        magnitude = self._compute_magnitude_spectrum(audio)

        # Frequency bins
        freqs = np.fft.rfftfreq(self.frame_length, 1 / self.sample_rate)

        # Compute cumulative energy
        total_energy = np.sum(magnitude, axis=0, keepdims=True)
        cumulative_energy = np.cumsum(magnitude, axis=0)

        # Find rolloff frequency for each frame
        rolloff = np.zeros(magnitude.shape[1], dtype=np.float32)
        threshold = percentile * total_energy

        for t in range(magnitude.shape[1]):
            # Find first bin where cumulative energy exceeds threshold
            idx = np.where(cumulative_energy[:, t] >= threshold[0, t])[0]
            if len(idx) > 0:
                rolloff[t] = freqs[idx[0]]
            else:
                rolloff[t] = freqs[-1]

        return rolloff

    def spectral_flux(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute spectral flux (change in spectrum over time).

        Spectral flux measures how quickly the spectrum is changing,
        useful for onset detection and identifying transients.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        flux : ndarray, shape (n_frames,)
            Spectral flux for each frame (first frame is zero)

        Examples
        --------
        >>> spectral = SpectralFeatures(sample_rate=16000)
        >>> flux = spectral.spectral_flux(audio)
        >>> # Find onsets (peaks in flux)
        >>> threshold = np.mean(flux) + 2 * np.std(flux)
        >>> onsets = np.where(flux > threshold)[0]
        >>> print(f"Found {len(onsets)} onsets")
        """
        validate_audio(audio)

        # Compute magnitude spectrum
        magnitude = self._compute_magnitude_spectrum(audio)

        # Compute flux: sum of squared differences
        flux = np.zeros(magnitude.shape[1], dtype=np.float32)

        for t in range(1, magnitude.shape[1]):
            diff = magnitude[:, t] - magnitude[:, t - 1]
            # Only consider increases (half-wave rectification)
            diff = np.maximum(0, diff)
            flux[t] = np.sum(diff**2)

        return flux

    def spectral_bandwidth(
        self, audio: npt.NDArray[np.float32], p: int = 2
    ) -> npt.NDArray[np.float32]:
        """
        Compute spectral bandwidth.

        Spectral bandwidth measures the spread of the spectrum around
        the centroid, indicating how concentrated or spread out the
        frequency content is.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal
        p : int, default=2
            Power for bandwidth calculation (2 = standard deviation)

        Returns
        -------
        bandwidth : ndarray, shape (n_frames,)
            Spectral bandwidth in Hz for each frame

        Examples
        --------
        >>> spectral = SpectralFeatures(sample_rate=16000)
        >>> bandwidth = spectral.spectral_bandwidth(audio)
        >>> print(f"Mean bandwidth: {np.mean(bandwidth):.1f} Hz")
        """
        validate_audio(audio)

        # Compute magnitude spectrum
        magnitude = self._compute_magnitude_spectrum(audio)

        # Frequency bins
        freqs = np.fft.rfftfreq(self.frame_length, 1 / self.sample_rate)

        # Compute centroid
        centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / (
            np.sum(magnitude, axis=0) + 1e-10
        )

        # Compute bandwidth: weighted standard deviation
        deviation = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
        bandwidth = np.sqrt(
            np.sum(deviation * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        )

        return bandwidth.astype(np.float32)

    def _compute_magnitude_spectrum(
        self, audio: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute magnitude spectrum using STFT.

        Parameters
        ----------
        audio : ndarray
            Input audio signal

        Returns
        -------
        magnitude : ndarray, shape (n_freqs, n_frames)
            Magnitude spectrum
        """
        # Compute STFT
        stft_matrix = stft(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Get magnitude
        magnitude = np.abs(stft_matrix)

        return magnitude.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"SpectralFeatures(sample_rate={self.sample_rate}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )
