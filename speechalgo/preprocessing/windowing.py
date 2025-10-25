"""
Window functions for speech signal processing.

Window functions are used to reduce spectral leakage when performing
Fourier analysis on finite-length signals. They taper the signal
smoothly to zero at the boundaries.

Mathematical Foundation
-----------------------
Window functions w[n] multiply the signal x[n]:
    y[n] = x[n] * w[n]

Common windows:
- Hamming: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
- Hanning: w[n] = 0.5 * (1 - cos(2πn / (N-1)))
- Blackman: w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))

References
----------
.. [1] Harris, F. J. (1978). "On the use of windows for harmonic analysis
       with the discrete Fourier transform". Proceedings of the IEEE, 66(1), 51-83.
.. [2] Oppenheim, A. V., & Schafer, R. W. (2009). "Discrete-time signal processing"
       (3rd ed.). Prentice Hall.
"""

import numpy as np
import numpy.typing as npt

from speechalgo.utils.validation import validate_frame_length


def hamming_window(length: int) -> npt.NDArray[np.float32]:
    """
    Generate a Hamming window.

    The Hamming window is widely used in speech processing for its
    good frequency resolution and moderate sidelobe suppression.

    Formula: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))

    Parameters
    ----------
    length : int
        Length of the window in samples

    Returns
    -------
    window : ndarray, shape (length,)
        Hamming window values

    Raises
    ------
    ValueError
        If length is not a positive integer

    Examples
    --------
    >>> window = hamming_window(512)
    >>> signal_windowed = signal * window

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(hamming_window(256))
    >>> plt.title('Hamming Window')
    >>> plt.show()

    Notes
    -----
    The Hamming window has a main lobe width of approximately 8π/N radians
    and maximum sidelobe level of -43 dB.
    """
    validate_frame_length(length)

    n = np.arange(length, dtype=np.float32)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))

    return window.astype(np.float32)


def hanning_window(length: int) -> npt.NDArray[np.float32]:
    """
    Generate a Hanning (Hann) window.

    The Hanning window provides better frequency resolution than rectangular
    windows while maintaining good sidelobe suppression. Also called Hann window.

    Formula: w[n] = 0.5 * (1 - cos(2πn / (N-1)))

    Parameters
    ----------
    length : int
        Length of the window in samples

    Returns
    -------
    window : ndarray, shape (length,)
        Hanning window values

    Raises
    ------
    ValueError
        If length is not a positive integer

    Examples
    --------
    >>> window = hanning_window(512)
    >>> spectrum = np.fft.fft(signal * window)

    >>> # Compare window shapes
    >>> hann = hanning_window(256)
    >>> hamm = hamming_window(256)
    >>> print(f"Hann center: {hann[128]:.3f}, Hamming center: {hamm[128]:.3f}")
    Hann center: 1.000, Hamming center: 1.000

    Notes
    -----
    The Hanning window has zero values at the endpoints (unlike Hamming),
    main lobe width of approximately 8π/N radians, and maximum sidelobe
    level of -32 dB.
    """
    validate_frame_length(length)

    n = np.arange(length, dtype=np.float32)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1)))

    return window.astype(np.float32)


def blackman_window(length: int) -> npt.NDArray[np.float32]:
    """
    Generate a Blackman window.

    The Blackman window provides excellent sidelobe suppression at the
    cost of wider main lobe compared to Hamming and Hanning windows.

    Formula: w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))

    Parameters
    ----------
    length : int
        Length of the window in samples

    Returns
    -------
    window : ndarray, shape (length,)
        Blackman window values

    Raises
    ------
    ValueError
        If length is not a positive integer

    Examples
    --------
    >>> window = blackman_window(512)
    >>> # Apply to signal for spectral analysis
    >>> windowed_signal = signal * window

    >>> # Blackman provides best sidelobe suppression
    >>> blackman = blackman_window(256)
    >>> print(f"Min value: {blackman.min():.6f}, Max value: {blackman.max():.6f}")
    Min value: 0.000000, Max value: 1.000000

    Notes
    -----
    The Blackman window has:
    - Main lobe width: approximately 12π/N radians (wider than Hamming/Hanning)
    - Maximum sidelobe level: -58 dB (better suppression)
    - Zero values at endpoints
    - Preferred when sidelobe suppression is critical
    """
    validate_frame_length(length)

    n = np.arange(length, dtype=np.float32)
    window = (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
        + 0.08 * np.cos(4.0 * np.pi * n / (length - 1))
    )

    return window.astype(np.float32)
