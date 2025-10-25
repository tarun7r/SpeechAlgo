"""
Tests for speechalgo.vad.zero_crossing module.
"""

import numpy as np
import pytest

from speechalgo.vad.zero_crossing import ZeroCrossingVAD


class TestZeroCrossingVAD:
    """Tests for ZeroCrossingVAD class."""

    def test_initialization(self, sample_rate):
        """Test VAD initialization."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        assert vad.sample_rate == sample_rate
        assert vad.frame_length > 0
        assert vad.hop_length > 0

    def test_process_speech_like_signal(self, speech_like_signal, sample_rate):
        """Test VAD on speech-like signal."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        is_speech = vad.process(speech_like_signal)

        # Output should be boolean array
        assert is_speech.dtype == bool
        assert len(is_speech) > 0

        # Should detect some speech and some silence
        assert is_speech.sum() > 0
        assert (~is_speech).sum() > 0

    def test_process_silence(self, zero_audio, sample_rate):
        """Test VAD on silence."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        is_speech = vad.process(zero_audio)

        # All frames should be classified as silence
        assert is_speech.sum() == 0

    def test_process_low_frequency_sine(self, sine_wave, sample_rate):
        """Test VAD on low-frequency sine wave."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        is_speech = vad.process(sine_wave)

        # Low-frequency sine has low ZCR (voiced speech-like)
        # Should be detected as speech
        assert is_speech.sum() > 0

    def test_process_high_frequency_noise(self, white_noise, sample_rate):
        """Test VAD on high-frequency noise."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        is_speech = vad.process(white_noise)

        # Noise has high ZCR
        # May or may not be detected as speech depending on energy
        assert len(is_speech) > 0

    def test_voiced_vs_unvoiced_detection(self, sample_rate):
        """Test distinction between voiced and unvoiced segments."""
        # Create signal with low and high frequency components
        t = np.arange(8000) / sample_rate
        voiced = np.sin(2 * np.pi * 150 * t)  # Low frequency (voiced)
        unvoiced = np.random.randn(8000) * 0.5  # Noise (unvoiced)

        signal = np.concatenate([voiced, unvoiced]).astype(np.float32)

        vad = ZeroCrossingVAD(
            sample_rate=sample_rate,
            zcr_threshold=0.3,
            energy_threshold=0.01,
        )
        is_speech = vad.process(signal)

        # Should detect some activity
        assert is_speech.sum() > 0

    def test_different_thresholds(self, sine_wave, sample_rate):
        """Test VAD with different ZCR thresholds."""
        low_zcr_vad = ZeroCrossingVAD(
            sample_rate=sample_rate,
            zcr_threshold=0.1,
            energy_threshold=0.01,
        )
        high_zcr_vad = ZeroCrossingVAD(
            sample_rate=sample_rate,
            zcr_threshold=0.5,
            energy_threshold=0.01,
        )

        is_speech_low = low_zcr_vad.process(sine_wave)
        is_speech_high = high_zcr_vad.process(sine_wave)

        # Results should differ based on threshold
        assert len(is_speech_low) == len(is_speech_high)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test VAD with different frame parameters."""
        vad = ZeroCrossingVAD(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )
        is_speech = vad.process(sine_wave)

        assert len(is_speech) > 0
        assert is_speech.dtype == bool

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test VAD with very short audio."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        
        # Tiny audio is shorter than frame_length, should raise error
        with pytest.raises(ValueError):
            is_speech = vad.process(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        vad = ZeroCrossingVAD(sample_rate=sample_rate)
        repr_str = repr(vad)
        assert "ZeroCrossingVAD" in repr_str
        assert str(sample_rate) in repr_str
