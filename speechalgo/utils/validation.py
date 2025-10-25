"""
Input validation utilities for consistent error checking.

This module provides validation functions to ensure inputs meet
the requirements of various algorithms.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt


def validate_audio(
    audio: npt.NDArray[np.float32],
    min_length: int = 1,
    max_channels: int = 1,
    name: str = "audio",
) -> None:
    """
    Validate audio signal array.

    Parameters
    ----------
    audio : ndarray
        Audio signal to validate
    min_length : int, default=1
        Minimum required length in samples
    max_channels : int, default=1
        Maximum allowed number of channels (1 for mono only)
    name : str, default='audio'
        Name of the variable for error messages

    Raises
    ------
    TypeError
        If audio is not a numpy array
    ValueError
        If audio shape or length is invalid

    Examples
    --------
    >>> audio = np.random.randn(16000)
    >>> validate_audio(audio, min_length=1024)  # OK

    >>> validate_audio(np.array([]), min_length=100)  # Raises ValueError
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(audio)}")

    if audio.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if audio.ndim > 2:
        raise ValueError(
            f"{name} must be 1D or 2D, got {audio.ndim}D array with shape {audio.shape}"
        )

    if audio.ndim == 2 and audio.shape[1] > max_channels:
        raise ValueError(f"{name} has {audio.shape[1]} channels, but max_channels={max_channels}")

    # Get effective length (minimum dimension for 2D arrays)
    effective_length = audio.shape[0] if audio.ndim == 1 else min(audio.shape)

    if effective_length < min_length:
        raise ValueError(f"{name} length ({effective_length}) is less than minimum ({min_length})")


def validate_sample_rate(sample_rate: int, min_rate: int = 1000, max_rate: int = 384000) -> None:
    """
    Validate sample rate value.

    Parameters
    ----------
    sample_rate : int
        Sample rate to validate in Hz
    min_rate : int, default=1000
        Minimum acceptable sample rate in Hz
    max_rate : int, default=384000
        Maximum acceptable sample rate in Hz

    Raises
    ------
    TypeError
        If sample_rate is not an integer
    ValueError
        If sample_rate is out of valid range

    Examples
    --------
    >>> validate_sample_rate(16000)  # OK
    >>> validate_sample_rate(8000)   # OK
    >>> validate_sample_rate(500)    # Raises ValueError
    """
    if not isinstance(sample_rate, int):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate)}")

    if sample_rate < min_rate or sample_rate > max_rate:
        raise ValueError(
            f"sample_rate ({sample_rate}) must be between {min_rate} and {max_rate} Hz"
        )


def validate_frame_length(frame_length: int, min_length: int = 16, max_length: int = 16384) -> None:
    """
    Validate frame length parameter.

    Parameters
    ----------
    frame_length : int
        Frame length to validate in samples
    min_length : int, default=16
        Minimum acceptable frame length
    max_length : int, default=16384
        Maximum acceptable frame length

    Raises
    ------
    TypeError
        If frame_length is not an integer
    ValueError
        If frame_length is out of valid range or not positive

    Examples
    --------
    >>> validate_frame_length(512)   # OK
    >>> validate_frame_length(1024)  # OK
    >>> validate_frame_length(0)     # Raises ValueError
    """
    if not isinstance(frame_length, int):
        raise TypeError(f"frame_length must be an integer, got {type(frame_length)}")

    if frame_length <= 0:
        raise ValueError(f"frame_length must be positive, got {frame_length}")

    if frame_length < min_length or frame_length > max_length:
        raise ValueError(
            f"frame_length ({frame_length}) must be between {min_length} and {max_length}"
        )


def validate_hop_length(hop_length: int, frame_length: Optional[int] = None) -> None:
    """
    Validate hop length parameter.

    Parameters
    ----------
    hop_length : int
        Hop length to validate in samples
    frame_length : int, optional
        Frame length for consistency check. If provided, hop_length
        should typically be <= frame_length

    Raises
    ------
    TypeError
        If hop_length is not an integer
    ValueError
        If hop_length is not positive or exceeds frame_length

    Examples
    --------
    >>> validate_hop_length(256, frame_length=512)  # OK
    >>> validate_hop_length(512, frame_length=512)  # OK
    >>> validate_hop_length(0)                       # Raises ValueError
    """
    if not isinstance(hop_length, int):
        raise TypeError(f"hop_length must be an integer, got {type(hop_length)}")

    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    if frame_length is not None and hop_length > frame_length:
        raise ValueError(
            f"hop_length ({hop_length}) should not exceed frame_length ({frame_length})"
        )


def validate_positive_float(value: float, name: str = "value") -> None:
    """
    Validate that a value is a positive float.

    Parameters
    ----------
    value : float
        Value to validate
    name : str, default='value'
        Name of the variable for error messages

    Raises
    ------
    TypeError
        If value is not numeric
    ValueError
        If value is not positive

    Examples
    --------
    >>> validate_positive_float(0.5, "threshold")  # OK
    >>> validate_positive_float(-0.1, "alpha")     # Raises ValueError
    """
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> None:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to validate
    min_val : float
        Minimum acceptable value (inclusive)
    max_val : float
        Maximum acceptable value (inclusive)
    name : str, default='value'
        Name of the variable for error messages

    Raises
    ------
    ValueError
        If value is outside the specified range

    Examples
    --------
    >>> validate_range(0.5, 0.0, 1.0, "alpha")  # OK
    >>> validate_range(1.5, 0.0, 1.0, "alpha")  # Raises ValueError
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{name} ({value}) must be between {min_val} and {max_val}")
