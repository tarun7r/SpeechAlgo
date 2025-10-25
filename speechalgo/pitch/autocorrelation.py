"""
Autocorrelation-based pitch detection.

Autocorrelation is one of the oldest and most fundamental methods for
pitch estimation. It exploits the periodic nature of voiced speech by
finding the lag at which the signal is most similar to itself.

Mathematical Foundation
-----------------------
The autocorrelation function (ACF) is defined as:

    R(τ) = Σ_{n=0}^{N-τ-1} x[n] * x[n+τ]

where:
- τ (tau) is the lag in samples
- N is the signal length
- x[n] is the signal

For a periodic signal with period T, the ACF has peaks at τ = T, 2T, 3T, ...

Pitch estimation algorithm:
1. Compute ACF for lags from τ_min to τ_max
2. Find the lag τ_max that maximizes R(τ)
3. F0 = sample_rate / τ_max

Improvements:
- Center clipping to reduce formant effects
- Normalization to account for amplitude decay
- Peak picking with threshold to avoid false detections

References
----------
.. [1] Rabiner, L. R. (1977). "On the use of autocorrelation analysis for
       pitch detection". IEEE Transactions on Acoustics, Speech, and Signal
       Processing, 25(1), 24-33.
.. [2] Duifhuis, H., Willems, L. F., & Sluyter, R. J. (1982). "Measurement
       of pitch in speech: An implementation of Goldstein's theory of pitch
       perception". The Journal of the Acoustical Society of America, 71(6).
"""

import numpy as np
import numpy.typing as npt

from speechalgo.pitch.base import BasePitchEstimator
from speechalgo.utils.validation import validate_audio


class Autocorrelation(BasePitchEstimator):
    """
    Estimate pitch using autocorrelation method.

    This classic method finds the periodicity in the signal by computing
    the autocorrelation function and locating its first significant peak.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    f0_min : float, default=80.0
        Minimum fundamental frequency to detect (Hz)
    f0_max : float, default=400.0
        Maximum fundamental frequency to detect (Hz)
    threshold : float, default=0.3
        Correlation threshold for voicing decision (0.0 to 1.0)
    center_clip : bool, default=True
        Apply center clipping to reduce formant influence

    Attributes
    ----------
    threshold : float
        Correlation threshold
    center_clip : bool
        Whether to apply center clipping

    Examples
    --------
    >>> # Estimate pitch from a single frame
    >>> pitch_estimator = Autocorrelation(sample_rate=16000)
    >>> frame = audio[0:512]
    >>> f0 = pitch_estimator.estimate(frame)
    >>> if f0 > 0:
    ...     print(f"Detected pitch: {f0:.1f} Hz")
    ... else:
    ...     print("Unvoiced or silence")

    >>> # Process multiple frames
    >>> from speechalgo.preprocessing import FrameExtractor
    >>> extractor = FrameExtractor(frame_length=512, hop_length=256)
    >>> frames = extractor.extract_frames(audio)
    >>> pitch_contour = [pitch_estimator.estimate(frame) for frame in frames]

    >>> # Visualize pitch contour
    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(len(pitch_contour)) * 256 / 16000
    >>> plt.plot(time, pitch_contour)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('F0 (Hz)')
    >>> plt.title('Pitch Contour')
    >>> plt.show()

    Notes
    -----
    Advantages:
    - Simple and intuitive
    - Computationally efficient
    - Works well for clean speech

    Limitations:
    - Sensitive to noise
    - May produce octave errors (pitch doubling/halving)
    - Formants can affect results
    - Requires careful threshold tuning

    For better accuracy, consider:
    - YIN algorithm (improved autocorrelation)
    - Cepstral method (handles harmonics better)
    - Combining multiple methods
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        threshold: float = 0.3,
        center_clip: bool = True,
    ):
        super().__init__(sample_rate, f0_min, f0_max)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.threshold = threshold
        self.center_clip = center_clip

        # Compute lag range
        self.lag_min = int(sample_rate / f0_max)
        self.lag_max = int(sample_rate / f0_min)

    def estimate(self, audio: npt.NDArray[np.float32]) -> float:
        """
        Estimate pitch from audio frame using autocorrelation.

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
        >>> estimator = Autocorrelation(sample_rate=16000)
        >>> frame = np.sin(2 * np.pi * 200 * np.arange(512) / 16000)
        >>> f0 = estimator.estimate(frame.astype(np.float32))
        >>> print(f"Estimated: {f0:.1f} Hz (expected: 200 Hz)")
        """
        validate_audio(audio, min_length=self.lag_max)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Apply center clipping if enabled
        if self.center_clip:
            audio = self._center_clip(audio)

        # Compute autocorrelation
        acf = self._compute_acf(audio)

        # Find the best lag (first significant peak)
        lag = self._find_best_lag(acf)

        if lag == 0:
            return 0.0

        # Convert lag to frequency
        f0 = self.sample_rate / lag

        return float(f0)

    def _center_clip(
        self, audio: npt.NDArray[np.float32], clip_level: float = 0.3
    ) -> npt.NDArray[np.float32]:
        """
        Apply center clipping to reduce formant influence.

        Center clipping zeros out samples below a threshold, which helps
        emphasize the fundamental frequency by reducing the effect of
        formant structure.

        Parameters
        ----------
        audio : ndarray
            Input signal
        clip_level : float
            Clipping level as fraction of max absolute value

        Returns
        -------
        clipped : ndarray
            Center-clipped signal
        """
        threshold = clip_level * np.max(np.abs(audio))
        clipped = np.copy(audio)
        mask = np.abs(clipped) < threshold
        clipped[mask] = 0.0
        return clipped

    def _compute_acf(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute autocorrelation function.

        Uses FFT-based computation for efficiency:
        ACF = IFFT(|FFT(x)|^2)

        Parameters
        ----------
        audio : ndarray
            Input signal

        Returns
        -------
        acf : ndarray, shape (lag_max+1,)
            Normalized autocorrelation values for valid lag range
        """
        n = len(audio)

        # Zero-pad to next power of 2 for efficient FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))

        # Compute ACF via FFT
        fft = np.fft.fft(audio, n=n_fft)
        power = np.abs(fft) ** 2
        acf_full = np.fft.ifft(power).real[:n]

        # Normalize by R(0)
        if acf_full[0] > 0:
            acf_full = acf_full / acf_full[0]

        # Extract relevant lag range
        acf = acf_full[self.lag_min : self.lag_max + 1]

        return acf.astype(np.float32)

    def _find_best_lag(self, acf: npt.NDArray[np.float32]) -> int:
        """
        Find the lag corresponding to the fundamental period.

        Finds the first significant peak in the ACF above the threshold.

        Parameters
        ----------
        acf : ndarray
            Autocorrelation function values

        Returns
        -------
        lag : int
            Best lag in samples (0 if no peak found)
        """
        if len(acf) == 0:
            return 0

        # Find peaks above threshold
        above_threshold = acf > self.threshold

        if not np.any(above_threshold):
            return 0

        # Find the first peak
        # A peak is a local maximum
        peaks = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and above_threshold[i]:
                peaks.append((i, acf[i]))

        if not peaks:
            # No clear peak found, use maximum value if above threshold
            max_idx = np.argmax(acf)
            if acf[max_idx] > self.threshold:
                return self.lag_min + max_idx
            return 0

        # Return the lag of the highest peak
        best_peak_idx, _ = max(peaks, key=lambda x: x[1])
        return self.lag_min + best_peak_idx

    def __repr__(self) -> str:
        return (
            f"Autocorrelation(sample_rate={self.sample_rate}, "
            f"f0_range=[{self.f0_min}, {self.f0_max}], "
            f"threshold={self.threshold})"
        )
