"""
Harmonic Product Spectrum (HPS) for pitch detection.

HPS is a frequency-domain pitch detection method that exploits the
harmonic structure of voiced speech. It identifies the fundamental
frequency by finding the frequency where harmonics align.

Mathematical Foundation
-----------------------
For a periodic signal with fundamental frequency F0, the spectrum
contains peaks at harmonics: F0, 2*F0, 3*F0, 4*F0, ...

HPS algorithm:
1. Compute magnitude spectrum: |X[k]|
2. Downsample spectrum by factors 2, 3, 4, ..., N:
   - X_2[k] = |X[2k]| (every 2nd bin)
   - X_3[k] = |X[3k]| (every 3rd bin)
   - X_4[k] = |X[4k]| (every 4th bin)
3. Multiply all downsampled spectra:
   HPS[k] = X[k] × X_2[k] × X_3[k] × X_4[k] × ...
4. Find peak in HPS:
   F0 = argmax(HPS[k])

The downsampling aligns all harmonics at the fundamental frequency,
creating a strong peak in the product spectrum.

Example:
- Original: peaks at 100, 200, 300, 400 Hz
- Downsample ÷2: peaks at 50, 100, 150, 200 Hz (100=200/2, 150=300/2, ...)
- Downsample ÷3: peaks at 33, 67, 100, 133 Hz
- Product: strongest peak at 100 Hz (F0)

References
----------
.. [1] Schroeder, M. R. (1968). "Period histogram and product spectrum:
       New methods for fundamental-frequency measurement". The Journal of
       the Acoustical Society of America, 43(4), 829-834.
.. [2] Medan, Y., Yair, E., & Chazan, D. (1991). "Super resolution pitch
       determination of speech signals". IEEE Transactions on Signal Processing.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.pitch.base import BasePitchEstimator
from speechalgo.utils.validation import validate_audio


class HPS(BasePitchEstimator):
    """
    Harmonic Product Spectrum pitch detection.

    HPS identifies the fundamental frequency by exploiting the harmonic
    structure of voiced speech. It is particularly effective for signals
    with clear harmonic content.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    f0_min : float, default=80.0
        Minimum fundamental frequency to detect (Hz)
    f0_max : float, default=400.0
        Maximum fundamental frequency to detect (Hz)
    n_harmonics : int, default=5
        Number of harmonics to use in the product (2-5 typical)
    n_fft : int, optional
        FFT size. If None, uses next power of 2 >= frame length

    Attributes
    ----------
    n_harmonics : int
        Number of harmonics used
    n_fft : int or None
        FFT size

    Examples
    --------
    >>> # HPS works well for harmonic sounds
    >>> hps = HPS(sample_rate=16000, n_harmonics=5)
    >>> frame = audio[0:512]
    >>> f0 = hps.estimate(frame)
    >>> print(f"Detected pitch: {f0:.1f} Hz")

    >>> # Compare with other methods
    >>> from speechalgo.pitch import YIN, Autocorrelation
    >>> yin = YIN(sample_rate=16000)
    >>> acf = Autocorrelation(sample_rate=16000)
    >>> hps_f0 = hps.estimate(frame)
    >>> yin_f0 = yin.estimate(frame)
    >>> acf_f0 = acf.estimate(frame)
    >>> print(f"HPS: {hps_f0:.1f}, YIN: {yin_f0:.1f}, ACF: {acf_f0:.1f}")

    >>> # Adjust number of harmonics
    >>> hps_3 = HPS(sample_rate=16000, n_harmonics=3)  # Faster
    >>> hps_7 = HPS(sample_rate=16000, n_harmonics=7)  # More accurate
    >>> f0_3 = hps_3.estimate(frame)
    >>> f0_7 = hps_7.estimate(frame)

    Notes
    -----
    Advantages:
    - Robust to missing fundamentals
    - Works well with harmonic sounds (speech, music)
    - Frequency-domain method (no lag computation)
    - Handles octave errors better than simple peak picking

    Limitations:
    - Requires clear harmonic structure
    - May fail with inharmonic sounds
    - Sensitive to noise between harmonics
    - Computationally more expensive than time-domain methods

    Recommended settings:
    - n_harmonics = 3-5 for speech
    - n_harmonics = 5-7 for music
    - Larger n_fft gives better frequency resolution

    The method is particularly effective when the fundamental is weak
    or missing, as it relies on the harmonic structure rather than
    detecting F0 directly.
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        n_harmonics: int = 5,
        n_fft: int = None,
    ):
        super().__init__(sample_rate, f0_min, f0_max)

        if n_harmonics < 2:
            raise ValueError(f"n_harmonics must be >= 2, got {n_harmonics}")

        self.n_harmonics = n_harmonics
        self.n_fft = n_fft

    def estimate(self, audio: npt.NDArray[np.float32]) -> float:
        """
        Estimate pitch using Harmonic Product Spectrum.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio frame (typically 20-40ms)

        Returns
        -------
        f0 : float
            Estimated fundamental frequency in Hz, or 0.0 if unvoiced

        Examples
        --------
        >>> hps = HPS(sample_rate=16000)
        >>> # Generate 200 Hz tone with harmonics
        >>> t = np.arange(512) / 16000
        >>> frame = (np.sin(2*np.pi*200*t) +
        ...          0.5*np.sin(2*np.pi*400*t) +
        ...          0.3*np.sin(2*np.pi*600*t)).astype(np.float32)
        >>> f0 = hps.estimate(frame)
        >>> print(f"Estimated: {f0:.1f} Hz (expected: 200 Hz)")
        """
        validate_audio(audio, min_length=64)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Compute magnitude spectrum
        magnitude_spectrum = self._compute_magnitude_spectrum(audio)

        # Compute HPS
        hps = self._compute_hps(magnitude_spectrum)

        # Find peak in valid frequency range
        f0 = self._find_peak_frequency(hps)

        return float(f0)

    def _compute_magnitude_spectrum(
        self, audio: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute magnitude spectrum.

        Parameters
        ----------
        audio : ndarray
            Input signal

        Returns
        -------
        magnitude : ndarray
            Magnitude spectrum (positive frequencies only)
        """
        # Determine FFT size
        if self.n_fft is None:
            n_fft = 2 ** int(np.ceil(np.log2(len(audio))))
        else:
            n_fft = self.n_fft

        # Compute FFT
        spectrum = np.fft.rfft(audio, n=n_fft)

        # Get magnitude
        magnitude = np.abs(spectrum)

        return magnitude.astype(np.float32)

    def _compute_hps(self, magnitude_spectrum: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute Harmonic Product Spectrum.

        Multiplies downsampled versions of the spectrum to emphasize
        the fundamental frequency.

        Parameters
        ----------
        magnitude_spectrum : ndarray
            Magnitude spectrum

        Returns
        -------
        hps : ndarray
            Harmonic product spectrum
        """
        # Start with the original spectrum
        hps = np.copy(magnitude_spectrum)

        # Multiply with downsampled versions
        for h in range(2, self.n_harmonics + 1):
            # Downsample by factor h
            downsampled_length = len(magnitude_spectrum) // h
            downsampled = magnitude_spectrum[::h][:downsampled_length]

            # Multiply (element-wise) with HPS, matching lengths
            min_length = min(len(hps), len(downsampled))
            hps[:min_length] *= downsampled[:min_length]

        return hps

    def _find_peak_frequency(self, hps: npt.NDArray[np.float32]) -> float:
        """
        Find the frequency with maximum HPS value.

        Searches in the valid frequency range for the peak.

        Parameters
        ----------
        hps : ndarray
            Harmonic product spectrum

        Returns
        -------
        f0 : float
            Frequency of the peak in Hz, or 0.0 if none found
        """
        # Convert frequency range to bin indices
        freq_resolution = self.sample_rate / (2 * (len(hps) - 1))
        bin_min = int(self.f0_min / freq_resolution)
        bin_max = int(self.f0_max / freq_resolution)

        # Ensure valid range
        bin_min = max(1, bin_min)  # Skip DC component
        bin_max = min(len(hps) - 1, bin_max)

        if bin_min >= bin_max:
            return 0.0

        # Extract relevant range
        hps_range = hps[bin_min : bin_max + 1]

        if len(hps_range) == 0 or np.max(hps_range) == 0:
            return 0.0

        # Find peak
        peak_idx = np.argmax(hps_range)
        peak_bin = bin_min + peak_idx

        # Convert bin to frequency
        f0 = peak_bin * freq_resolution

        return f0

    def __repr__(self) -> str:
        return (
            f"HPS(sample_rate={self.sample_rate}, "
            f"f0_range=[{self.f0_min}, {self.f0_max}], "
            f"n_harmonics={self.n_harmonics})"
        )
