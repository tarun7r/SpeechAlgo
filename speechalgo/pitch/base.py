"""
Abstract base class for pitch estimation algorithms.

This module defines the interface that all pitch detection implementations
must follow, ensuring consistent API across different methods.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt


class BasePitchEstimator(ABC):
    """
    Abstract base class for pitch estimation algorithms.

    All pitch detection implementations should inherit from this class
    and implement the `estimate` method.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    f0_min : float, default=80.0
        Minimum fundamental frequency to detect (Hz)
    f0_max : float, default=400.0
        Maximum fundamental frequency to detect (Hz)

    Attributes
    ----------
    sample_rate : int
        Sample rate in Hz
    f0_min : float
        Minimum F0 in Hz
    f0_max : float
        Maximum F0 in Hz

    Notes
    -----
    Typical F0 ranges:
    - Male speech: 85-180 Hz
    - Female speech: 165-255 Hz
    - Children: 250-400 Hz
    - General speech: 80-400 Hz (default)
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
    ):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        if f0_min <= 0:
            raise ValueError(f"f0_min must be positive, got {f0_min}")

        if f0_max <= f0_min:
            raise ValueError(f"f0_max ({f0_max}) must be greater than f0_min ({f0_min})")

        if f0_max > sample_rate / 2:
            raise ValueError(f"f0_max ({f0_max}) cannot exceed Nyquist frequency ({sample_rate/2})")

        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max

    @abstractmethod
    def estimate(self, audio: npt.NDArray[np.float32]) -> Union[float, npt.NDArray[np.float32]]:
        """
        Estimate fundamental frequency (pitch) from audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,) or (n_frames, frame_length)
            Input audio signal or frames

        Returns
        -------
        f0 : float or ndarray
            Estimated fundamental frequency in Hz.
            Returns 0.0 or array of zeros if no pitch detected (unvoiced/silence).

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement estimate()")

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the pitch estimator."""
        raise NotImplementedError
