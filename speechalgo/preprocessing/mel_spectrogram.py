"""
Mel-scale spectrogram computation.

A mel-spectrogram is a time-frequency representation where the frequency
axis is mapped to the mel scale, which approximates human auditory perception.
This is a fundamental representation for speech and audio analysis.

Mathematical Foundation
-----------------------
1. Compute STFT: X[k,n] = FFT(x[n] * w[n])
2. Compute power spectrogram: P[k,n] = |X[k,n]|^2
3. Apply mel filterbank: M[m,n] = Σ_k (F[m,k] * P[k,n])

where F[m,k] is the mel filterbank matrix mapping FFT bins to mel bins.

Mel scale conversion:
    mel = 2595 * log10(1 + hz / 700)
    hz = 700 * (10^(mel / 2595) - 1)

References
----------
.. [1] Stevens, S. S., Volkmann, J., & Newman, E. B. (1937).
       "A scale for the measurement of the psychological magnitude pitch"
.. [2] Davis, S., & Mermelstein, P. (1980). "Comparison of parametric
       representations for monosyllabic word recognition"
.. [3] O'Shaughnessy, D. (1987). "Speech communication: human and machine"
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from speechalgo.utils.signal_processing import create_mel_filterbank, stft
from speechalgo.utils.validation import (
    validate_audio,
    validate_frame_length,
    validate_sample_rate,
)


class MelSpectrogram:
    """
    Compute mel-scale spectrogram from audio signals.

    The mel-spectrogram provides a perceptually-motivated time-frequency
    representation useful for speech recognition, speaker identification,
    emotion recognition, and audio classification tasks.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    n_fft : int, default=512
        FFT size (number of frequency bins)
    hop_length : int, default=256
        Number of samples between successive frames
    n_mels : int, default=40
        Number of mel filterbanks
    freq_min : float, default=0.0
        Minimum frequency in Hz
    freq_max : float, optional
        Maximum frequency in Hz. If None, uses sample_rate / 2
    window : str, default='hann'
        Window function: 'hann', 'hamming', 'blackman', 'boxcar'
    power : float, default=2.0
        Exponent for magnitude spectrogram (1.0 = magnitude, 2.0 = power)

    Attributes
    ----------
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT size
    hop_length : int
        Hop length in samples
    n_mels : int
        Number of mel bins
    mel_filterbank : ndarray, shape (n_mels, n_fft // 2 + 1)
        Mel filterbank matrix

    Examples
    --------
    >>> # Compute mel-spectrogram
    >>> mel_spec = MelSpectrogram(sample_rate=16000, n_mels=40)
    >>> mel_s = mel_spec.process(audio)
    >>> print(mel_s.shape)  # (n_mels, n_frames)
    (40, 63)

    >>> # Convert to log scale (decibels)
    >>> log_mel = mel_spec.to_db(mel_s)

    >>> # Visualize
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
    >>> plt.ylabel('Mel frequency bin')
    >>> plt.xlabel('Time frame')
    >>> plt.colorbar(label='dB')
    >>> plt.show()

    Notes
    -----
    Typical parameters for speech:
    - sample_rate: 16000 Hz
    - n_fft: 512 (32ms at 16kHz)
    - hop_length: 256 (16ms at 16kHz, 50% overlap)
    - n_mels: 40-80 (40 common for ASR, 80 for music)
    - freq_min: 0 Hz or 80 Hz (remove very low frequencies)
    - freq_max: 8000 Hz (Nyquist frequency for 16kHz)
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        window: str = "hann",
        power: float = 2.0,
    ):
        validate_sample_rate(sample_rate)
        validate_frame_length(n_fft)

        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")

        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = window
        self.power = power

        # Create mel filterbank
        self.mel_filterbank = create_mel_filterbank(
            n_filters=n_mels,
            n_fft=n_fft,
            sample_rate=sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute mel-spectrogram from audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        mel_spectrogram : ndarray, shape (n_mels, n_frames)
            Mel-scale spectrogram

        Examples
        --------
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> mel_spec = MelSpectrogram(sample_rate=16000)
        >>> mel_s = mel_spec.process(audio)
        >>> print(mel_s.shape)
        (40, 63)
        """
        validate_audio(audio)

        # Compute STFT
        stft_matrix = stft(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            n_fft=self.n_fft,
        )

        # Compute power/magnitude spectrogram
        magnitude = np.abs(stft_matrix)
        power_spec = magnitude**self.power

        # Apply mel filterbank
        mel_spec = self.mel_filterbank @ power_spec

        return mel_spec.astype(np.float32)

    def to_db(
        self,
        mel_spectrogram: npt.NDArray[np.float32],
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = 80.0,
    ) -> npt.NDArray[np.float32]:
        """
        Convert mel-spectrogram to decibel (dB) scale.

        Parameters
        ----------
        mel_spectrogram : ndarray, shape (n_mels, n_frames)
            Mel-spectrogram in linear scale
        ref : float, default=1.0
            Reference value for dB calculation
        amin : float, default=1e-10
            Minimum threshold to avoid log(0)
        top_db : float, optional
            Maximum dB value (clips lower values). If None, no clipping.

        Returns
        -------
        mel_db : ndarray, shape (n_mels, n_frames)
            Mel-spectrogram in decibel scale

        Examples
        --------
        >>> mel_s = mel_spec.process(audio)
        >>> mel_db = mel_spec.to_db(mel_s)
        >>> print(f"Range: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")
        Range: [-80.0, 45.3] dB
        """
        # Ensure non-negative and above minimum
        mel_spectrogram = np.maximum(amin, mel_spectrogram)

        # Convert to dB
        mel_db = 10.0 * np.log10(mel_spectrogram / ref)

        # Clip to top_db if specified
        if top_db is not None:
            mel_db = np.maximum(mel_db, mel_db.max() - top_db)

        return mel_db.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"MelSpectrogram(sample_rate={self.sample_rate}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
            f"n_mels={self.n_mels})"
        )
