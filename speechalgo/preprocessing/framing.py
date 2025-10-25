"""
Frame extraction with overlapping windows.

Framing divides a continuous audio signal into short, overlapping segments
for time-frequency analysis. This is fundamental to most speech processing
algorithms as speech is quasi-stationary over short time periods (10-30ms).

Mathematical Foundation
-----------------------
Given signal x[n] of length N, frames are extracted as:
    frame_k[m] = x[k * hop + m] * w[m]

where:
- k is the frame index
- hop is the hop length (stride between frames)
- w[m] is the window function
- m ∈ [0, frame_length)

The overlap is: overlap = frame_length - hop_length

References
----------
.. [1] Rabiner, L., & Juang, B. H. (1993). "Fundamentals of speech recognition".
       Prentice Hall.
.. [2] Oppenheim, A. V., & Schafer, R. W. (2009). "Discrete-time signal processing".
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.utils.validation import (
    validate_audio,
    validate_frame_length,
    validate_hop_length,
)


class FrameExtractor:
    """
    Extract overlapping frames from an audio signal.

    This class provides methods to split an audio signal into short,
    overlapping frames for analysis. Frames can optionally be windowed
    to reduce spectral leakage.

    Parameters
    ----------
    frame_length : int
        Length of each frame in samples (typically 256-512 for speech)
    hop_length : int
        Number of samples to advance between frames (stride).
        Smaller hop_length gives more time resolution but more frames.
    window : str or ndarray, optional
        Window function to apply. Options:
        - 'hamming': Hamming window (default)
        - 'hanning': Hanning window
        - 'blackman': Blackman window
        - None: No windowing (rectangular)
        - ndarray: Custom window of length frame_length
    pad_mode : str, default='reflect'
        Padding mode for edges. Options: 'reflect', 'constant', 'edge'

    Attributes
    ----------
    frame_length : int
        Frame length in samples
    hop_length : int
        Hop length in samples
    window_array : ndarray or None
        Window function values

    Examples
    --------
    >>> # Extract frames with 50% overlap
    >>> extractor = FrameExtractor(frame_length=512, hop_length=256)
    >>> frames = extractor.extract_frames(audio)
    >>> print(frames.shape)  # (n_frames, 512)

    >>> # No windowing
    >>> extractor = FrameExtractor(frame_length=512, hop_length=512, window=None)
    >>> frames = extractor.extract_frames(audio)

    >>> # Process each frame
    >>> for frame in frames:
    ...     spectrum = np.fft.rfft(frame)
    ...     # Analyze spectrum...

    Notes
    -----
    For speech analysis, typical parameters are:
    - frame_length: 20-30ms (320-480 samples at 16kHz)
    - hop_length: 10-15ms (50-75% overlap)
    - window: Hamming or Hanning
    """

    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        window: Optional[str] = "hamming",
        pad_mode: str = "reflect",
    ):
        validate_frame_length(frame_length)
        validate_hop_length(hop_length, frame_length)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_mode = pad_mode

        # Create window function
        if window is None:
            self.window_array = None
        elif isinstance(window, str):
            self.window_array = self._create_window(window, frame_length)
        elif isinstance(window, np.ndarray):
            if len(window) != frame_length:
                raise ValueError(
                    f"Window length ({len(window)}) must match frame_length ({frame_length})"
                )
            self.window_array = window.astype(np.float32)
        else:
            raise ValueError(f"Invalid window type: {type(window)}")

    def extract_frames(
        self, audio: npt.NDArray[np.float32], pad: bool = True
    ) -> npt.NDArray[np.float32]:
        """
        Extract frames from audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal
        pad : bool, default=True
            If True, pad signal to ensure complete coverage

        Returns
        -------
        frames : ndarray, shape (n_frames, frame_length)
            Extracted frames, windowed if window was specified

        Examples
        --------
        >>> audio = np.random.randn(16000)
        >>> extractor = FrameExtractor(512, 256)
        >>> frames = extractor.extract_frames(audio)
        >>> print(frames.shape)
        (63, 512)
        """
        validate_audio(audio, min_length=self.frame_length)

        if audio.ndim > 1:
            audio = audio.flatten()

        # Pad signal if requested
        if pad:
            audio = self._pad_signal(audio)

        # Calculate number of frames
        n_frames = 1 + (len(audio) - self.frame_length) // self.hop_length

        # Extract frames using stride tricks for efficiency
        frames = self._extract_frames_strided(audio, n_frames)

        # Apply window if specified
        if self.window_array is not None:
            frames = frames * self.window_array

        return frames

    def frame_count(self, audio_length: int, pad: bool = True) -> int:
        """
        Calculate the number of frames for a given audio length.

        Parameters
        ----------
        audio_length : int
            Length of audio signal in samples
        pad : bool, default=True
            Whether padding will be applied

        Returns
        -------
        n_frames : int
            Number of frames that will be extracted

        Examples
        --------
        >>> extractor = FrameExtractor(512, 256)
        >>> n_frames = extractor.frame_count(16000)
        >>> print(n_frames)
        63
        """
        if pad:
            # Account for padding
            pad_length = self.frame_length // 2
            audio_length = audio_length + 2 * pad_length

        if audio_length < self.frame_length:
            return 0

        return 1 + (audio_length - self.frame_length) // self.hop_length

    def _create_window(self, window: str, length: int) -> npt.NDArray[np.float32]:
        """Create a window function."""
        if window == "hamming":
            from speechalgo.preprocessing.windowing import hamming_window

            return hamming_window(length)
        elif window == "hanning":
            from speechalgo.preprocessing.windowing import hanning_window

            return hanning_window(length)
        elif window == "blackman":
            from speechalgo.preprocessing.windowing import blackman_window

            return blackman_window(length)
        else:
            raise ValueError(f"Unknown window type: {window}")

    def _pad_signal(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Pad signal for complete frame coverage."""
        pad_length = self.frame_length // 2

        if self.pad_mode == "reflect":
            audio = np.pad(audio, (pad_length, pad_length), mode="reflect")
        elif self.pad_mode == "constant":
            audio = np.pad(audio, (pad_length, pad_length), mode="constant")
        elif self.pad_mode == "edge":
            audio = np.pad(audio, (pad_length, pad_length), mode="edge")
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

        return audio

    def _extract_frames_strided(
        self, audio: npt.NDArray[np.float32], n_frames: int
    ) -> npt.NDArray[np.float32]:
        """Extract frames using numpy stride tricks for efficiency."""
        # Create a view into the audio array with strided access
        shape = (n_frames, self.frame_length)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])

        frames = np.lib.stride_tricks.as_strided(
            audio, shape=shape, strides=strides, writeable=False
        )

        # Return a copy to avoid issues with the view
        return frames.copy()
