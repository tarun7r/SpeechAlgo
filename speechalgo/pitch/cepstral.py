"""
Cepstral pitch detection.

The cepstrum is the inverse Fourier transform of the logarithm of the
power spectrum. It provides a way to separate the source (periodic excitation)
from the filter (vocal tract resonances), making it useful for pitch detection.

Mathematical Foundation
-----------------------
The real cepstrum is defined as:

    c[n] = IFFT(log(|FFT(x[n])|^2))

Or equivalently:
    c[n] = IFFT(log(|X[k]|))

where:
- FFT is the Fast Fourier Transform
- IFFT is the Inverse FFT
- |X[k]| is the magnitude spectrum

The independent variable of the cepstrum is called "quefrency" (anagram of
frequency), measured in samples or time.

For periodic signals, the cepstrum exhibits a peak at the quefrency
corresponding to the fundamental period. This peak is located at:

    quefrency = sample_rate / F0

The cepstrum effectively separates:
- Low quefrencies: Spectral envelope (vocal tract)
- High quefrencies: Fine spectral structure (periodicity)

Pitch detection algorithm:
1. Compute power spectrum
2. Take logarithm
3. Apply inverse FFT to get cepstrum
4. Search for peak in quefrency range
5. F0 = sample_rate / quefrency_peak

References
----------
.. [1] Noll, A. M. (1967). "Cepstrum pitch determination". The Journal of
       the Acoustical Society of America, 41(2), 293-309.
.. [2] Oppenheim, A. V., & Schafer, R. W. (2004). "From frequency to quefrency:
       A history of the cepstrum". IEEE Signal Processing Magazine, 21(5), 95-106.
.. [3] Rabiner, L. R., & Schafer, R. W. (2007). "Introduction to digital
       speech processing". Foundations and Trends in Signal Processing, 1(1-2).
"""

import numpy as np
import numpy.typing as npt

from speechalgo.pitch.base import BasePitchEstimator
from speechalgo.utils.validation import validate_audio


class CepstralPitch(BasePitchEstimator):
    """
    Estimate pitch using cepstral analysis.

    The cepstrum separates the excitation (pitch) from the vocal tract
    filter, enabling robust pitch detection especially in the presence
    of formant structure.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    f0_min : float, default=80.0
        Minimum fundamental frequency to detect (Hz)
    f0_max : float, default=400.0
        Maximum fundamental frequency to detect (Hz)
    n_fft : int, optional
        FFT size. If None, uses next power of 2 >= frame length
    lifter_cutoff : int, default=80
        Liftering cutoff in quefrency samples. Sets to zero all cepstral
        coefficients below this to remove vocal tract influence.

    Attributes
    ----------
    n_fft : int or None
        FFT size
    lifter_cutoff : int
        Liftering cutoff quefrency

    Examples
    --------
    >>> # Cepstral method is robust to formants
    >>> cepstral = CepstralPitch(sample_rate=16000)
    >>> frame = audio[0:512]
    >>> f0 = cepstral.estimate(frame)
    >>> print(f"Detected pitch: {f0:.1f} Hz")

    >>> # Compare with autocorrelation
    >>> from speechalgo.pitch import Autocorrelation
    >>> acf = Autocorrelation(sample_rate=16000)
    >>> cepstral_f0 = cepstral.estimate(frame)
    >>> acf_f0 = acf.estimate(frame)
    >>> print(f"Cepstral: {cepstral_f0:.1f}, ACF: {acf_f0:.1f}")

    >>> # Process speech with strong formants
    >>> vowel_frame = audio_vowel[0:512]
    >>> f0 = cepstral.estimate(vowel_frame)
    >>> print(f"F0 despite formants: {f0:.1f} Hz")

    Notes
    -----
    Advantages:
    - Separates pitch from formant structure
    - Robust to spectral envelope variations
    - Good for speech with strong formants
    - Handles harmonic complex tones well

    Limitations:
    - Requires longer frames (512-1024 samples)
    - Computationally more expensive (two FFTs)
    - Sensitive to spectral zeros (low energy regions)
    - May have reduced time resolution

    The liftering operation zeros out low quefrency components to
    remove the influence of the vocal tract envelope, leaving only
    the periodic excitation information.

    Typical lifter cutoff: 1-5 ms (16-80 samples at 16kHz)
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        n_fft: int = None,
        lifter_cutoff: int = 80,
    ):
        super().__init__(sample_rate, f0_min, f0_max)

        if lifter_cutoff < 0:
            raise ValueError(f"lifter_cutoff must be non-negative, got {lifter_cutoff}")

        self.n_fft = n_fft
        self.lifter_cutoff = lifter_cutoff

        # Compute quefrency range (in samples)
        self.quefrency_min = int(sample_rate / f0_max)
        self.quefrency_max = int(sample_rate / f0_min)

    def estimate(self, audio: npt.NDArray[np.float32]) -> float:
        """
        Estimate pitch using cepstral analysis.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio frame (typically 30-60ms)

        Returns
        -------
        f0 : float
            Estimated fundamental frequency in Hz, or 0.0 if unvoiced

        Examples
        --------
        >>> cepstral = CepstralPitch(sample_rate=16000)
        >>> frame = np.sin(2 * np.pi * 150 * np.arange(1024) / 16000)
        >>> f0 = cepstral.estimate(frame.astype(np.float32))
        >>> print(f"Estimated: {f0:.1f} Hz (expected: 150 Hz)")
        """
        validate_audio(audio, min_length=self.quefrency_max)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Compute cepstrum
        cepstrum = self._compute_cepstrum(audio)

        # Apply liftering to remove low quefrencies
        cepstrum = self._apply_lifter(cepstrum)

        # Find peak in quefrency range
        quefrency = self._find_peak_quefrency(cepstrum)

        if quefrency == 0:
            return 0.0

        # Convert quefrency to frequency
        f0 = self.sample_rate / quefrency

        return float(f0)

    def _compute_cepstrum(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute the real cepstrum.

        c[n] = IFFT(log(|FFT(x[n])|))

        Parameters
        ----------
        audio : ndarray
            Input signal

        Returns
        -------
        cepstrum : ndarray
            Real cepstrum values
        """
        # Determine FFT size
        if self.n_fft is None:
            n_fft = 2 ** int(np.ceil(np.log2(len(audio))))
        else:
            n_fft = self.n_fft

        # Compute FFT
        spectrum = np.fft.fft(audio, n=n_fft)

        # Compute magnitude spectrum
        magnitude = np.abs(spectrum)

        # Add small constant to avoid log(0)
        magnitude = np.maximum(magnitude, 1e-10)

        # Take logarithm
        log_magnitude = np.log(magnitude)

        # Compute cepstrum via inverse FFT
        cepstrum = np.fft.ifft(log_magnitude).real

        return cepstrum.astype(np.float32)

    def _apply_lifter(self, cepstrum: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply liftering to remove low quefrency components.

        Liftering zeros out cepstral coefficients below the cutoff,
        removing the vocal tract envelope and leaving the excitation.

        Parameters
        ----------
        cepstrum : ndarray
            Input cepstrum

        Returns
        -------
        liftered : ndarray
            Liftered cepstrum
        """
        liftered = np.copy(cepstrum)

        # Zero out low quefrencies (vocal tract envelope)
        if self.lifter_cutoff > 0:
            liftered[: self.lifter_cutoff] = 0

        return liftered

    def _find_peak_quefrency(self, cepstrum: npt.NDArray[np.float32]) -> int:
        """
        Find the quefrency with maximum cepstral value.

        Searches in the valid quefrency range for the peak.

        Parameters
        ----------
        cepstrum : ndarray
            Cepstrum values

        Returns
        -------
        quefrency : int
            Quefrency of the peak in samples, or 0 if none found
        """
        # Ensure valid range
        q_min = max(self.quefrency_min, self.lifter_cutoff)
        q_max = min(self.quefrency_max, len(cepstrum) - 1)

        if q_min >= q_max:
            return 0

        # Extract relevant quefrency range
        cepstrum_range = cepstrum[q_min : q_max + 1]

        if len(cepstrum_range) == 0:
            return 0

        # Find peak
        peak_idx = np.argmax(cepstrum_range)
        quefrency = q_min + peak_idx

        # Check if peak is significant
        # (cepstrum value should be positive for periodic signals)
        if cepstrum[quefrency] <= 0:
            return 0

        return quefrency

    def __repr__(self) -> str:
        return (
            f"CepstralPitch(sample_rate={self.sample_rate}, "
            f"f0_range=[{self.f0_min}, {self.f0_max}], "
            f"lifter_cutoff={self.lifter_cutoff})"
        )
