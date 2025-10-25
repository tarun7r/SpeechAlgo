"""
Pytest configuration and shared fixtures for SpeechAlgo tests.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    """Standard sample rate for testing."""
    return 16000


@pytest.fixture
def duration():
    """Standard duration for test signals in seconds."""
    return 1.0


@pytest.fixture
def sine_wave(sample_rate, duration):
    """Generate a sine wave signal."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    frequency = 440.0  # A4 note
    signal = np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


@pytest.fixture
def chirp_signal(sample_rate, duration):
    """Generate a chirp signal (frequency sweep)."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    f0, f1 = 100.0, 1000.0
    signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    return signal.astype(np.float32)


@pytest.fixture
def white_noise(sample_rate, duration):
    """Generate white noise."""
    n_samples = int(sample_rate * duration)
    signal = np.random.randn(n_samples)
    return signal.astype(np.float32)


@pytest.fixture
def noisy_sine(sine_wave, white_noise):
    """Generate a sine wave with additive white noise (SNR ~10dB)."""
    noise_level = 0.1
    return sine_wave + noise_level * white_noise


@pytest.fixture
def speech_like_signal(sample_rate, duration):
    """
    Generate a speech-like signal with voiced and unvoiced segments.

    Alternates between:
    - Voiced segments (low-frequency periodic)
    - Unvoiced segments (high-frequency noise)
    """
    n_samples = int(sample_rate * duration)
    signal = np.zeros(n_samples, dtype=np.float32)

    # Segment duration
    segment_samples = n_samples // 4

    # Voiced segment 1 (0-25%)
    t1 = np.arange(segment_samples) / sample_rate
    signal[:segment_samples] = np.sin(2 * np.pi * 120 * t1)

    # Unvoiced segment 1 (25-50%)
    signal[segment_samples : 2 * segment_samples] = 0.3 * np.random.randn(segment_samples)

    # Voiced segment 2 (50-75%)
    t2 = np.arange(segment_samples) / sample_rate
    signal[2 * segment_samples : 3 * segment_samples] = np.sin(2 * np.pi * 200 * t2)

    # Silence (75-100%)
    signal[3 * segment_samples :] = 0.01 * np.random.randn(n_samples - 3 * segment_samples)

    return signal.astype(np.float32)


@pytest.fixture
def frame_length():
    """Standard frame length for testing."""
    return 512


@pytest.fixture
def hop_length():
    """Standard hop length for testing."""
    return 256


@pytest.fixture
def n_fft():
    """Standard FFT size for testing."""
    return 512


@pytest.fixture
def n_mels():
    """Standard number of mel bands for testing."""
    return 40


@pytest.fixture
def n_mfcc():
    """Standard number of MFCC coefficients for testing."""
    return 13


@pytest.fixture
def tiny_audio():
    """Tiny audio signal for edge case testing."""
    return np.array([0.1, -0.2, 0.3, -0.1], dtype=np.float32)


@pytest.fixture
def zero_audio():
    """Zero audio signal for edge case testing."""
    return np.zeros(1000, dtype=np.float32)


@pytest.fixture
def single_sample():
    """Single sample audio for edge case testing."""
    return np.array([0.5], dtype=np.float32)
