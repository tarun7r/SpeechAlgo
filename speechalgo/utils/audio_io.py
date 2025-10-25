"""
Audio I/O utilities for loading and saving audio files.

This module provides simple wrappers around soundfile for consistent
audio file handling throughout the library.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import soundfile as sf


def load_audio(
    filepath: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    dtype: str = "float32",
) -> Tuple[npt.NDArray[np.float32], int]:
    """
    Load an audio file from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the audio file to load
    sample_rate : int, optional
        Target sample rate. If provided and different from file sample rate,
        the audio will be resampled (requires scipy). If None, uses the
        file's native sample rate.
    mono : bool, default=True
        If True, convert stereo audio to mono by averaging channels
    dtype : str, default='float32'
        Output data type

    Returns
    -------
    audio : ndarray, shape (n_samples,) or (n_samples, n_channels)
        Loaded audio data normalized to [-1.0, 1.0]
    sr : int
        Sample rate of the loaded audio

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist
    ValueError
        If the file format is not supported

    Examples
    --------
    >>> audio, sr = load_audio('speech.wav')
    >>> print(f"Loaded {len(audio)} samples at {sr} Hz")
    Loaded 48000 samples at 16000 Hz

    >>> stereo_audio, sr = load_audio('music.wav', mono=False)
    >>> print(stereo_audio.shape)
    (48000, 2)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        audio, sr = sf.read(str(filepath), dtype=dtype)
    except Exception as e:
        raise ValueError(f"Failed to load audio file {filepath}: {e}")

    # Convert to mono if requested
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sample_rate is not None and sample_rate != sr:
        audio = _resample(audio, sr, sample_rate)
        sr = sample_rate

    return audio, sr


def save_audio(
    filepath: Union[str, Path],
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    subtype: str = "PCM_16",
) -> None:
    """
    Save audio data to a file.

    Parameters
    ----------
    filepath : str or Path
        Path where the audio file will be saved
    audio : ndarray
        Audio data to save, shape (n_samples,) or (n_samples, n_channels)
    sample_rate : int
        Sample rate of the audio data
    subtype : str, default='PCM_16'
        Audio encoding subtype. Common values:
        - 'PCM_16': 16-bit PCM
        - 'PCM_24': 24-bit PCM
        - 'FLOAT': 32-bit float
        - 'DOUBLE': 64-bit float

    Raises
    ------
    ValueError
        If audio data or sample rate is invalid

    Examples
    --------
    >>> audio = np.random.randn(16000).astype(np.float32)
    >>> save_audio('output.wav', audio, 16000)

    >>> stereo = np.random.randn(16000, 2).astype(np.float32)
    >>> save_audio('stereo.wav', stereo, 44100, subtype='PCM_24')
    """
    filepath = Path(filepath)

    if audio.size == 0:
        raise ValueError("Cannot save empty audio array")

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    # Ensure audio is in valid range for integer formats
    if subtype.startswith("PCM"):
        audio = np.clip(audio, -1.0, 1.0)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(filepath), audio, sample_rate, subtype=subtype)
    except Exception as e:
        raise ValueError(f"Failed to save audio file {filepath}: {e}")


def _resample(
    audio: npt.NDArray[np.float32], orig_sr: int, target_sr: int
) -> npt.NDArray[np.float32]:
    """
    Resample audio to a target sample rate using scipy.

    Parameters
    ----------
    audio : ndarray
        Input audio signal
    orig_sr : int
        Original sample rate
    target_sr : int
        Target sample rate

    Returns
    -------
    resampled : ndarray
        Resampled audio signal
    """
    from scipy import signal

    if orig_sr == target_sr:
        return audio

    # Calculate the number of samples in the resampled signal
    num_samples = int(len(audio) * target_sr / orig_sr)

    # Use scipy's resample function
    resampled = signal.resample(audio, num_samples)

    return resampled.astype(np.float32)
