"""
Abstract base class for Voice Activity Detection algorithms.

This module defines the interface that all VAD implementations must follow,
ensuring consistent API across different detection methods.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseVAD(ABC):
    """
    Abstract base class for Voice Activity Detection.

    All VAD implementations should inherit from this class and implement
    the `process` method.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz

    Attributes
    ----------
    sample_rate : int
        Sample rate in Hz
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @abstractmethod
    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """
        Detect voice activity in audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        vad_labels : ndarray, shape (n_frames,)
            Boolean array where True indicates voice activity

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement process()")

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the VAD detector."""
        raise NotImplementedError
