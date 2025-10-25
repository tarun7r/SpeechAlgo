"""
SpeechAlgo: A Comprehensive Speech Processing Library

SpeechAlgo provides clean, educational implementations of fundamental speech
processing algorithms organized into five categories:

- Preprocessing: Windowing, framing, pre-emphasis, MFCC, mel-spectrogram
- Voice Activity Detection (VAD): Energy-based, spectral entropy, zero-crossing
- Pitch Detection: Autocorrelation, YIN, cepstral, harmonic product spectrum
- Speech Enhancement: Spectral subtraction, Wiener filtering, noise gate
- Feature Extraction: Spectral features, temporal features, delta features

Example usage:
    >>> from speechalgo.preprocessing import hamming_window, MFCC
    >>> from speechalgo.vad import EnergyBasedVAD
    >>> from speechalgo.pitch import YIN
    >>> from speechalgo.enhancement import WienerFilter

    >>> # Create a window
    >>> window = hamming_window(512)

    >>> # Extract MFCC features
    >>> mfcc = MFCC(sample_rate=16000, n_mfcc=13)
    >>> features = mfcc.process(audio_signal)

    >>> # Initialize VAD detector
    >>> vad = EnergyBasedVAD(sample_rate=16000)
    >>> is_speech = vad.process(audio_signal)

    >>> # Estimate pitch
    >>> pitch_detector = YIN(sample_rate=16000)
    >>> pitch = pitch_detector.estimate(audio_frame)

    >>> # Enhance audio
    >>> enhancer = WienerFilter(sample_rate=16000)
    >>> enhancer.estimate_noise(noisy_audio[:16000])
    >>> clean_audio = enhancer.process(noisy_audio)
"""

from speechalgo.version import __version__

# Preprocessing
from speechalgo.preprocessing.windowing import (
    hamming_window,
    hanning_window,
    blackman_window,
)
from speechalgo.preprocessing.framing import FrameExtractor
from speechalgo.preprocessing.pre_emphasis import PreEmphasis
from speechalgo.preprocessing.mfcc import MFCC
from speechalgo.preprocessing.mel_spectrogram import MelSpectrogram

# Voice Activity Detection
from speechalgo.vad.energy_based import EnergyBasedVAD
from speechalgo.vad.spectral_entropy import SpectralEntropyVAD
from speechalgo.vad.zero_crossing import ZeroCrossingVAD

# Pitch Detection
from speechalgo.pitch.autocorrelation import Autocorrelation
from speechalgo.pitch.yin import YIN
from speechalgo.pitch.cepstral import CepstralPitch
from speechalgo.pitch.hps import HPS

# Speech Enhancement
from speechalgo.enhancement.spectral_subtraction import SpectralSubtraction
from speechalgo.enhancement.wiener_filter import WienerFilter
from speechalgo.enhancement.noise_gate import NoiseGate

# Feature Extraction
from speechalgo.features.spectral import SpectralFeatures
from speechalgo.features.temporal import TemporalFeatures
from speechalgo.features.delta import DeltaFeatures

__all__ = [
    # Version
    "__version__",
    # Preprocessing
    "hamming_window",
    "hanning_window",
    "blackman_window",
    "FrameExtractor",
    "PreEmphasis",
    "MFCC",
    "MelSpectrogram",
    # VAD
    "EnergyBasedVAD",
    "SpectralEntropyVAD",
    "ZeroCrossingVAD",
    # Pitch
    "Autocorrelation",
    "YIN",
    "CepstralPitch",
    "HPS",
    # Enhancement
    "SpectralSubtraction",
    "WienerFilter",
    "NoiseGate",
    # Features
    "SpectralFeatures",
    "TemporalFeatures",
    "DeltaFeatures",
]

