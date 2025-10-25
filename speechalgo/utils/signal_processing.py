"""
Signal processing utilities including FFT, STFT, and filtering operations.

This module provides foundational signal processing operations used
throughout the library.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import signal


def apply_fft(
    frame: npt.NDArray[np.float32], n_fft: Optional[int] = None
) -> npt.NDArray[np.complex128]:
    """
    Apply Fast Fourier Transform to a signal frame.

    Parameters
    ----------
    frame : ndarray, shape (n_samples,)
        Input signal frame
    n_fft : int, optional
        FFT size. If None, uses the next power of 2 >= len(frame)

    Returns
    -------
    spectrum : ndarray, shape (n_fft,)
        Complex-valued FFT output

    Examples
    --------
    >>> frame = np.random.randn(512)
    >>> spectrum = apply_fft(frame)
    >>> magnitude = np.abs(spectrum)
    >>> phase = np.angle(spectrum)
    """
    if n_fft is None:
        n_fft = _next_power_of_2(len(frame))

    return np.fft.fft(frame, n=n_fft)


def stft(
    audio: npt.NDArray[np.float32],
    frame_length: int = 512,
    hop_length: int = 256,
    window: str = "hann",
    n_fft: Optional[int] = None,
) -> npt.NDArray[np.complex128]:
    """
    Compute Short-Time Fourier Transform.

    The STFT divides the signal into overlapping frames and computes
    the FFT of each frame, producing a time-frequency representation.

    Parameters
    ----------
    audio : ndarray, shape (n_samples,)
        Input audio signal
    frame_length : int, default=512
        Length of each frame in samples
    hop_length : int, default=256
        Number of samples between successive frames
    window : str, default='hann'
        Window function name. Options: 'hann', 'hamming', 'blackman', 'boxcar'
    n_fft : int, optional
        FFT size. If None, uses frame_length

    Returns
    -------
    stft_matrix : ndarray, shape (n_fft//2 + 1, n_frames)
        Complex-valued STFT matrix (only positive frequencies)

    Examples
    --------
    >>> audio = np.random.randn(16000)
    >>> S = stft(audio, frame_length=512, hop_length=256)
    >>> magnitude = np.abs(S)
    >>> power = magnitude ** 2
    """
    if n_fft is None:
        n_fft = frame_length

    # Get window function
    window_array = _get_window(window, frame_length)

    # Use scipy's stft implementation
    frequencies, times, stft_matrix = signal.stft(
        audio,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
        nfft=n_fft,
        window=window_array,
        boundary=None,
        padded=False,
    )

    return stft_matrix


def istft(
    stft_matrix: npt.NDArray[np.complex128],
    hop_length: int = 256,
    window: str = "hann",
    frame_length: Optional[int] = None,
) -> npt.NDArray[np.float32]:
    """
    Compute Inverse Short-Time Fourier Transform.

    Reconstructs a time-domain signal from its STFT representation.

    Parameters
    ----------
    stft_matrix : ndarray, shape (n_fft//2 + 1, n_frames)
        Complex-valued STFT matrix
    hop_length : int, default=256
        Number of samples between successive frames
    window : str, default='hann'
        Window function name (should match forward STFT)
    frame_length : int, optional
        Length of each frame. If None, inferred from stft_matrix

    Returns
    -------
    audio : ndarray
        Reconstructed time-domain signal

    Examples
    --------
    >>> S = stft(audio, frame_length=512, hop_length=256)
    >>> reconstructed = istft(S, hop_length=256)
    >>> np.allclose(audio[:len(reconstructed)], reconstructed, atol=1e-5)
    True
    """
    if frame_length is None:
        frame_length = 2 * (stft_matrix.shape[0] - 1)

    window_array = _get_window(window, frame_length)

    # Use scipy's istft implementation
    _, audio = signal.istft(
        stft_matrix,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
        window=window_array,
        boundary=False,
    )

    return audio.astype(np.float32)


def hz_to_mel(frequencies: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert frequencies from Hz to mel scale.

    The mel scale is a perceptual scale of pitches judged by listeners
    to be equal in distance from one another.

    Formula: mel = 2595 * log10(1 + hz / 700)

    Parameters
    ----------
    frequencies : ndarray
        Frequencies in Hz

    Returns
    -------
    mels : ndarray
        Frequencies in mel scale

    References
    ----------
    .. [1] Stevens, S. S., Volkmann, J., & Newman, E. B. (1937).
           "A scale for the measurement of the psychological magnitude pitch"
    """
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert frequencies from mel scale to Hz.

    Formula: hz = 700 * (10^(mel / 2595) - 1)

    Parameters
    ----------
    mels : ndarray
        Frequencies in mel scale

    Returns
    -------
    frequencies : ndarray
        Frequencies in Hz
    """
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def create_mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    freq_min: float = 0.0,
    freq_max: Optional[float] = None,
) -> npt.NDArray[np.float32]:
    """
    Create a mel-scale filterbank.

    Parameters
    ----------
    n_filters : int
        Number of mel filters
    n_fft : int
        FFT size
    sample_rate : int
        Sample rate of the audio signal
    freq_min : float, default=0.0
        Minimum frequency (Hz)
    freq_max : float, optional
        Maximum frequency (Hz). If None, uses sample_rate / 2

    Returns
    -------
    filterbank : ndarray, shape (n_filters, n_fft // 2 + 1)
        Mel filterbank matrix

    References
    ----------
    .. [1] Davis, S., & Mermelstein, P. (1980). "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences"
    """
    if freq_max is None:
        freq_max = sample_rate / 2.0

    # Convert to mel scale
    mel_min = hz_to_mel(np.array([freq_min]))[0]
    mel_max = hz_to_mel(np.array([freq_max]))[0]

    # Create evenly spaced mel points
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)

    # Convert back to Hz
    hz_points = mel_to_hz(mel_points)

    # Convert Hz to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))

    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank.astype(np.float32)


def _get_window(window: str, length: int) -> npt.NDArray[np.float32]:
    """
    Get a window function.

    Parameters
    ----------
    window : str
        Window type: 'hann', 'hamming', 'blackman', 'boxcar'
    length : int
        Window length

    Returns
    -------
    window_array : ndarray
        Window function values
    """
    if window == "hann":
        return signal.windows.hann(length, sym=False).astype(np.float32)
    elif window == "hamming":
        return signal.windows.hamming(length, sym=False).astype(np.float32)
    elif window == "blackman":
        return signal.windows.blackman(length, sym=False).astype(np.float32)
    elif window == "boxcar":
        return np.ones(length, dtype=np.float32)
    else:
        raise ValueError(f"Unknown window type: {window}")


def _next_power_of_2(x: int) -> int:
    """
    Calculate the next power of 2 greater than or equal to x.

    Parameters
    ----------
    x : int
        Input value

    Returns
    -------
    power : int
        Next power of 2
    """
    return 1 << (x - 1).bit_length()
