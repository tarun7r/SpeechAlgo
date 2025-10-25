"""
Pre-emphasis filter for high-frequency boosting.

Pre-emphasis is a first-order high-pass filter that amplifies higher
frequencies in the speech signal. This compensates for the natural
spectral tilt of speech (energy decreases with frequency) and balances
the frequency spectrum, improving the performance of subsequent analysis.

Mathematical Foundation
-----------------------
The pre-emphasis filter is defined as:
    y[n] = x[n] - α * x[n-1]

where α is the pre-emphasis coefficient (typically 0.95-0.97).

In the z-domain:
    H(z) = 1 - α * z^(-1)

This is a first-order FIR filter that boosts high frequencies.

The inverse operation (de-emphasis) is:
    x[n] = y[n] + α * x[n-1]
    H(z) = 1 / (1 - α * z^(-1))

References
----------
.. [1] Rabiner, L. R., & Schafer, R. W. (2007). "Introduction to digital
       speech processing". Foundations and Trends in Signal Processing, 1(1), 1-194.
.. [2] Furui, S. (1986). "Speaker-independent isolated word recognition using
       dynamic features of speech spectrum". IEEE Transactions on Acoustics,
       Speech, and Signal Processing, 34(1), 52-59.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.utils.validation import validate_audio, validate_range


class PreEmphasis:
    """
    Apply pre-emphasis filtering to speech signals.

    Pre-emphasis amplifies high frequencies to balance the spectrum
    and improve the signal-to-noise ratio in subsequent processing.

    Parameters
    ----------
    coefficient : float, default=0.97
        Pre-emphasis coefficient α. Typical range: 0.95-0.97
        Higher values give more high-frequency boost.

    Attributes
    ----------
    coefficient : float
        The pre-emphasis coefficient

    Examples
    --------
    >>> # Apply pre-emphasis with default coefficient
    >>> pre_emp = PreEmphasis(coefficient=0.97)
    >>> emphasized = pre_emp.process(audio)

    >>> # Compare spectra before and after
    >>> import matplotlib.pyplot as plt
    >>> original_fft = np.abs(np.fft.rfft(audio[:512]))
    >>> emphasized_fft = np.abs(np.fft.rfft(emphasized[:512]))
    >>> plt.plot(original_fft, label='Original')
    >>> plt.plot(emphasized_fft, label='Pre-emphasized')
    >>> plt.legend()
    >>> plt.show()

    >>> # Process and then reverse
    >>> emphasized = pre_emp.process(audio)
    >>> recovered = pre_emp.inverse(emphasized)
    >>> np.allclose(audio[1:], recovered[1:], atol=1e-6)
    True

    Notes
    -----
    Pre-emphasis is typically applied before:
    - MFCC extraction
    - LPC analysis
    - Pitch detection
    - Formant analysis

    Common coefficient values:
    - α = 0.95: Moderate boost
    - α = 0.97: Strong boost (most common)
    - α = 0.99: Very strong boost

    The first sample is left unchanged as it has no previous sample.
    """

    def __init__(self, coefficient: float = 0.97):
        validate_range(coefficient, 0.0, 1.0, "coefficient")
        self.coefficient = coefficient

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply pre-emphasis filter to audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        emphasized : ndarray, shape (n_samples,)
            Pre-emphasized audio signal

        Examples
        --------
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> pre_emp = PreEmphasis(0.97)
        >>> emphasized = pre_emp.process(audio)
        >>> print(emphasized.shape)
        (16000,)
        """
        validate_audio(audio)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Apply filter: y[n] = x[n] - α * x[n-1]
        emphasized = np.copy(audio)
        emphasized[1:] = audio[1:] - self.coefficient * audio[:-1]

        return emphasized.astype(np.float32)

    def inverse(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply de-emphasis (inverse pre-emphasis) to recover original signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Pre-emphasized audio signal

        Returns
        -------
        de_emphasized : ndarray, shape (n_samples,)
            De-emphasized (recovered) audio signal

        Examples
        --------
        >>> pre_emp = PreEmphasis(0.97)
        >>> emphasized = pre_emp.process(audio)
        >>> recovered = pre_emp.inverse(emphasized)
        >>> # First sample may differ slightly, rest should match
        >>> np.allclose(audio[1:], recovered[1:], atol=1e-6)
        True

        Notes
        -----
        De-emphasis is the inverse operation: x[n] = y[n] + α * x[n-1]
        This is a recursive filter (IIR) unlike the pre-emphasis (FIR).
        """
        validate_audio(audio)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Apply inverse filter: x[n] = y[n] + α * x[n-1]
        de_emphasized = np.copy(audio)
        for i in range(1, len(audio)):
            de_emphasized[i] = audio[i] + self.coefficient * de_emphasized[i - 1]

        return de_emphasized.astype(np.float32)

    def __repr__(self) -> str:
        return f"PreEmphasis(coefficient={self.coefficient})"
