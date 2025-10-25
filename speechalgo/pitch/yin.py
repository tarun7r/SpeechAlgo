"""
YIN algorithm for pitch detection.

YIN is an improved autocorrelation-based pitch detection algorithm that
reduces errors common in traditional autocorrelation methods. It is one
of the most accurate and widely-used pitch detection algorithms.

Mathematical Foundation
-----------------------
YIN uses a difference function instead of autocorrelation:

    d(τ) = Σ_{n=0}^{N-τ-1} (x[n] - x[n+τ])^2

This can be expanded to:
    d(τ) = R(0) + R(0) - 2*R(τ)

where R(τ) is the autocorrelation function.

Key innovations:
1. Cumulative mean normalized difference function (CMNDF):
   d'(τ) = d(τ) / [(1/τ) * Σ_{j=1}^{τ} d(j)]

2. Absolute threshold for period detection
3. Parabolic interpolation for sub-sample accuracy

The algorithm finds the smallest τ where d'(τ) < threshold.

Advantages over autocorrelation:
- Better at avoiding octave errors
- More robust to noise
- Fewer false positives
- Sub-sample accuracy via interpolation

References
----------
.. [1] de Cheveigné, A., & Kawahara, H. (2002). "YIN, a fundamental frequency
       estimator for speech and music". The Journal of the Acoustical Society
       of America, 111(4), 1917-1930.
.. [2] Cheveigné, A. D., & Kawahara, H. (2002). "YIN, a fundamental frequency
       estimator for speech and music". JASA, 111(4), 1917-1930.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.pitch.base import BasePitchEstimator
from speechalgo.utils.validation import validate_audio


class YIN(BasePitchEstimator):
    """
    YIN algorithm for robust pitch estimation.

    YIN is an improved autocorrelation method that provides better accuracy
    and robustness compared to traditional autocorrelation, especially for
    avoiding octave errors.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    f0_min : float, default=80.0
        Minimum fundamental frequency to detect (Hz)
    f0_max : float, default=400.0
        Maximum fundamental frequency to detect (Hz)
    threshold : float, default=0.1
        Threshold for absolute threshold (lower = stricter, 0.1-0.15 typical)
    interpolate : bool, default=True
        Use parabolic interpolation for sub-sample accuracy

    Attributes
    ----------
    threshold : float
        Absolute threshold for period detection
    interpolate : bool
        Whether to use parabolic interpolation

    Examples
    --------
    >>> # YIN is recommended for speech pitch detection
    >>> yin = YIN(sample_rate=16000, f0_min=80, f0_max=400)
    >>> frame = audio[0:512]
    >>> f0 = yin.estimate(frame)
    >>> print(f"Detected pitch: {f0:.1f} Hz")

    >>> # Process entire audio
    >>> from speechalgo.preprocessing import FrameExtractor
    >>> extractor = FrameExtractor(frame_length=512, hop_length=256)
    >>> frames = extractor.extract_frames(audio)
    >>> pitch_contour = np.array([yin.estimate(frame) for frame in frames])

    >>> # YIN is more accurate than autocorrelation
    >>> from speechalgo.pitch import Autocorrelation
    >>> acf = Autocorrelation(sample_rate=16000)
    >>> yin_result = yin.estimate(frame)
    >>> acf_result = acf.estimate(frame)
    >>> print(f"YIN: {yin_result:.1f} Hz, ACF: {acf_result:.1f} Hz")

    Notes
    -----
    YIN advantages:
    - Superior accuracy compared to autocorrelation
    - Fewer octave errors (pitch doubling/halving)
    - Better performance in noisy conditions
    - Sub-sample accuracy with interpolation
    - Well-tested and widely used in research

    Recommended for:
    - Speech pitch tracking
    - Musical instrument pitch detection
    - Prosody analysis
    - Intonation modeling

    Typical threshold values:
    - 0.10: Strict (fewer false positives)
    - 0.15: Balanced (recommended)
    - 0.20: Lenient (more detections, may include errors)
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        threshold: float = 0.1,
        interpolate: bool = True,
    ):
        super().__init__(sample_rate, f0_min, f0_max)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.threshold = threshold
        self.interpolate = interpolate

        # Compute lag range (tau range)
        self.tau_min = int(sample_rate / f0_max)
        self.tau_max = int(sample_rate / f0_min)

    def estimate(self, audio: npt.NDArray[np.float32]) -> float:
        """
        Estimate pitch using YIN algorithm.

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
        >>> yin = YIN(sample_rate=16000)
        >>> frame = np.sin(2 * np.pi * 200 * np.arange(512) / 16000)
        >>> f0 = yin.estimate(frame.astype(np.float32))
        >>> print(f"Estimated: {f0:.1f} Hz (expected: 200 Hz)")
        """
        validate_audio(audio, min_length=self.tau_max)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Step 1: Compute difference function
        df = self._difference_function(audio)

        # Step 2: Compute cumulative mean normalized difference function
        cmndf = self._cumulative_mean_normalized_difference(df)

        # Step 3: Absolute threshold
        tau = self._absolute_threshold(cmndf)

        if tau == 0:
            return 0.0

        # Step 4: Parabolic interpolation (optional)
        if self.interpolate:
            tau = self._parabolic_interpolation(cmndf, tau)

        # Convert tau to frequency
        f0 = self.sample_rate / tau

        return float(f0)

    def _difference_function(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute the difference function.

        d(τ) = Σ_{n=0}^{N-τ-1} (x[n] - x[n+τ])^2

        This is efficiently computed using autocorrelation:
        d(τ) = 2 * (R(0) - R(τ))

        Parameters
        ----------
        audio : ndarray
            Input signal

        Returns
        -------
        df : ndarray, shape (tau_max+1,)
            Difference function values
        """
        n = len(audio)
        df = np.zeros(self.tau_max + 1, dtype=np.float32)

        # Use FFT-based autocorrelation for efficiency
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft = np.fft.fft(audio, n=n_fft)
        power = np.abs(fft) ** 2
        acf = np.fft.ifft(power).real[:n]

        # Compute energy terms for normalization
        energy = np.cumsum(audio**2)
        energy = np.concatenate(([0], energy))

        # Compute difference function
        for tau in range(self.tau_max + 1):
            if tau == 0:
                df[tau] = 0
            else:
                # d(tau) = energy[0:N-tau] + energy[tau:N] - 2*R(tau)
                df[tau] = energy[n] + energy[n - tau] - 2 * acf[tau]

        return df

    def _cumulative_mean_normalized_difference(
        self, df: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute cumulative mean normalized difference function (CMNDF).

        d'(τ) = d(τ) / [(1/τ) * Σ_{j=1}^{τ} d(j)]
             = d(τ) / (mean of d[1:τ+1])

        This normalization makes the threshold independent of signal level.

        Parameters
        ----------
        df : ndarray
            Difference function

        Returns
        -------
        cmndf : ndarray
            Cumulative mean normalized difference function
        """
        cmndf = np.zeros_like(df)
        cmndf[0] = 1.0  # By definition

        # Compute cumulative sum for efficient mean calculation
        cumsum = np.cumsum(df[1:])

        for tau in range(1, len(df)):
            if cumsum[tau - 1] == 0:
                cmndf[tau] = 1.0
            else:
                cmndf[tau] = df[tau] / (cumsum[tau - 1] / tau)

        return cmndf

    def _absolute_threshold(self, cmndf: npt.NDArray[np.float32]) -> int:
        """
        Find the smallest tau where cmndf(tau) < threshold.

        This is the "absolute threshold" step in YIN.

        Parameters
        ----------
        cmndf : ndarray
            Cumulative mean normalized difference function

        Returns
        -------
        tau : int
            Best lag (period) in samples, or 0 if none found
        """
        # Start search from tau_min
        for tau in range(self.tau_min, self.tau_max + 1):
            if cmndf[tau] < self.threshold:
                # Found a candidate, but check for local minimum
                while tau + 1 < self.tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return tau

        # No period found below threshold
        return 0

    def _parabolic_interpolation(self, cmndf: npt.NDArray[np.float32], tau: int) -> float:
        """
        Refine tau estimate using parabolic interpolation.

        Fits a parabola through three points: (tau-1, tau, tau+1)
        and finds the minimum.

        Parameters
        ----------
        cmndf : ndarray
            Cumulative mean normalized difference function
        tau : int
            Initial tau estimate

        Returns
        -------
        tau_refined : float
            Refined tau with sub-sample accuracy
        """
        if tau == 0 or tau >= len(cmndf) - 1:
            return float(tau)

        # Get three points
        x0, x1, x2 = tau - 1, tau, tau + 1
        y0, y1, y2 = cmndf[x0], cmndf[x1], cmndf[x2]

        # Fit parabola and find minimum
        # Minimum is at: x = x1 + 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        denominator = y0 - 2 * y1 + y2

        if abs(denominator) < 1e-10:
            return float(tau)

        offset = 0.5 * (y0 - y2) / denominator
        tau_refined = tau + offset

        # Ensure result is in valid range
        tau_refined = np.clip(tau_refined, self.tau_min, self.tau_max)

        return float(tau_refined)

    def __repr__(self) -> str:
        return (
            f"YIN(sample_rate={self.sample_rate}, "
            f"f0_range=[{self.f0_min}, {self.f0_max}], "
            f"threshold={self.threshold})"
        )
