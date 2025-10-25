"""
Mel-Frequency Cepstral Coefficients (MFCC) extraction.

MFCCs are one of the most widely used feature representations in speech
and audio processing. They provide a compact representation of the spectral
envelope that closely models human auditory perception.

Mathematical Foundation
-----------------------
MFCC computation involves these steps:

1. Pre-emphasis (optional): y[n] = x[n] - α*x[n-1]
2. Frame the signal: Divide into overlapping frames
3. Apply window: w[n] * frame[n]
4. Compute FFT: X[k] = FFT(windowed_frame)
5. Compute power spectrum: P[k] = |X[k]|^2
6. Apply mel filterbank: M[m] = Σ_k (F[m,k] * P[k])
7. Take logarithm: L[m] = log(M[m])
8. Apply DCT: MFCC[n] = DCT(L[m])

The DCT decorrelates the log mel-spectrogram coefficients and compresses
energy into the lower coefficients.

DCT-II formula:
    C[n] = Σ_{m=0}^{M-1} L[m] * cos(πn(m + 0.5) / M)

References
----------
.. [1] Davis, S., & Mermelstein, P. (1980). "Comparison of parametric
       representations for monosyllabic word recognition in continuously
       spoken sentences". IEEE Transactions on Acoustics, Speech, and
       Signal Processing, 28(4), 357-366.
.. [2] Rabiner, L., & Juang, B. H. (1993). "Fundamentals of speech recognition".
.. [3] Lyons, J. (2012). "Mel Frequency Cepstral Coefficient (MFCC) tutorial"
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.fftpack import dct

from speechalgo.preprocessing.mel_spectrogram import MelSpectrogram
from speechalgo.preprocessing.pre_emphasis import PreEmphasis
from speechalgo.utils.validation import validate_audio, validate_sample_rate


class MFCC:
    """
    Extract Mel-Frequency Cepstral Coefficients from audio signals.

    MFCCs are the standard acoustic features for speech recognition,
    speaker identification, and various audio classification tasks.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signal in Hz
    n_mfcc : int, default=13
        Number of MFCC coefficients to extract
    n_fft : int, default=512
        FFT size
    hop_length : int, default=256
        Number of samples between successive frames
    n_mels : int, default=40
        Number of mel filterbanks
    freq_min : float, default=0.0
        Minimum frequency in Hz
    freq_max : float, optional
        Maximum frequency in Hz. If None, uses sample_rate / 2
    window : str, default='hann'
        Window function
    use_energy : bool, default=True
        If True, replace first MFCC with log energy
    lifter : int, optional
        Cepstral liftering coefficient. If provided, applies sinusoidal
        liftering to MFCCs to weight higher coefficients.
    pre_emphasis : float, optional
        Pre-emphasis coefficient (0.95-0.97). If None, no pre-emphasis.

    Attributes
    ----------
    n_mfcc : int
        Number of MFCC coefficients
    mel_spectrogram : MelSpectrogram
        Mel-spectrogram extractor
    pre_emphasis_filter : PreEmphasis or None
        Pre-emphasis filter if enabled

    Examples
    --------
    >>> # Extract 13 MFCCs
    >>> mfcc_extractor = MFCC(sample_rate=16000, n_mfcc=13)
    >>> mfcc_features = mfcc_extractor.process(audio)
    >>> print(mfcc_features.shape)  # (13, n_frames)
    (13, 63)

    >>> # With pre-emphasis and energy replacement
    >>> mfcc_extractor = MFCC(
    ...     sample_rate=16000,
    ...     n_mfcc=13,
    ...     pre_emphasis=0.97,
    ...     use_energy=True
    ... )
    >>> mfcc_features = mfcc_extractor.process(audio)

    >>> # Visualize MFCCs
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(mfcc_features, aspect='auto', origin='lower', cmap='coolwarm')
    >>> plt.ylabel('MFCC coefficient')
    >>> plt.xlabel('Time frame')
    >>> plt.colorbar()
    >>> plt.show()

    Notes
    -----
    Standard MFCC configuration for speech recognition:
    - n_mfcc: 13 (12 cepstral + 1 energy)
    - n_mels: 40
    - sample_rate: 16000 Hz
    - n_fft: 512 (32ms)
    - hop_length: 256 (16ms, 50% overlap)
    - pre_emphasis: 0.97
    - use_energy: True

    The first coefficient (C0) represents the average log energy.
    When use_energy=True, it's replaced with the frame's log energy.

    Coefficients 1-12 capture the spectral shape (formants, envelope).
    Higher coefficients capture fine spectral details.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        window: str = "hann",
        use_energy: bool = True,
        lifter: Optional[int] = None,
        pre_emphasis: Optional[float] = None,
    ):
        validate_sample_rate(sample_rate)

        if n_mfcc <= 0:
            raise ValueError(f"n_mfcc must be positive, got {n_mfcc}")

        if n_mfcc > n_mels:
            raise ValueError(f"n_mfcc ({n_mfcc}) cannot exceed n_mels ({n_mels})")

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_energy = use_energy
        self.lifter = lifter

        # Create mel-spectrogram extractor
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            freq_min=freq_min,
            freq_max=freq_max,
            window=window,
            power=2.0,
        )

        # Create pre-emphasis filter if specified
        if pre_emphasis is not None:
            self.pre_emphasis_filter = PreEmphasis(coefficient=pre_emphasis)
        else:
            self.pre_emphasis_filter = None

        # Create lifter weights if specified
        if lifter is not None:
            self.lifter_weights = self._create_lifter(n_mfcc, lifter)
        else:
            self.lifter_weights = None

    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Extract MFCC features from audio signal.

        Parameters
        ----------
        audio : ndarray, shape (n_samples,)
            Input audio signal

        Returns
        -------
        mfcc : ndarray, shape (n_mfcc, n_frames)
            MFCC feature matrix

        Examples
        --------
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> mfcc_extractor = MFCC(sample_rate=16000)
        >>> mfcc_features = mfcc_extractor.process(audio)
        >>> print(mfcc_features.shape)
        (13, 63)
        """
        validate_audio(audio)

        # Apply pre-emphasis if enabled
        if self.pre_emphasis_filter is not None:
            audio = self.pre_emphasis_filter.process(audio)

        # Store original audio for energy computation if needed
        original_audio = audio if self.use_energy else None

        # Compute mel-spectrogram
        mel_spec = self.mel_spectrogram.process(audio)

        # Add small constant to avoid log(0)
        mel_spec = np.maximum(mel_spec, 1e-10)

        # Take logarithm
        log_mel_spec = np.log(mel_spec)

        # Apply DCT
        mfcc = dct(log_mel_spec, type=2, axis=0, norm="ortho")[: self.n_mfcc, :]

        # Apply liftering if specified
        if self.lifter_weights is not None:
            mfcc = mfcc * self.lifter_weights[:, np.newaxis]

        # Replace first coefficient with log energy if requested
        if self.use_energy and original_audio is not None:
            energy = self._compute_frame_energy(original_audio)
            if energy.shape[0] == mfcc.shape[1]:
                mfcc[0, :] = energy

        return mfcc.astype(np.float32)

    def _compute_frame_energy(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute log energy for each frame.

        Parameters
        ----------
        audio : ndarray
            Audio signal

        Returns
        -------
        log_energy : ndarray, shape (n_frames,)
            Log energy per frame
        """
        from speechalgo.preprocessing.framing import FrameExtractor

        # Extract frames using the same parameters
        frame_extractor = FrameExtractor(
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            window=None,  # No windowing for energy computation
        )

        frames = frame_extractor.extract_frames(audio)

        # Compute energy per frame
        energy = np.sum(frames**2, axis=1)

        # Avoid log(0)
        energy = np.maximum(energy, 1e-10)

        # Take logarithm
        log_energy = np.log(energy)

        return log_energy

    def _create_lifter(self, n_mfcc: int, lifter: int) -> npt.NDArray[np.float32]:
        """
        Create sinusoidal liftering weights.

        Liftering emphasizes mid-range cepstral coefficients while
        de-emphasizing very low and very high coefficients.

        Formula: lift[n] = 1 + (lifter / 2) * sin(πn / lifter)

        Parameters
        ----------
        n_mfcc : int
            Number of MFCC coefficients
        lifter : int
            Liftering parameter (typically 22)

        Returns
        -------
        weights : ndarray, shape (n_mfcc,)
            Liftering weights
        """
        n = np.arange(n_mfcc)
        weights = 1 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
        return weights.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"MFCC(sample_rate={self.sample_rate}, n_mfcc={self.n_mfcc}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length})"
        )
