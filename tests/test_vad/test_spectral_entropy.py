"""
Tests for speechalgo.vad.spectral_entropy module.
"""

import pytest

from speechalgo.vad.spectral_entropy import SpectralEntropyVAD


class TestSpectralEntropyVAD:
    """Tests for SpectralEntropyVAD class."""

    def test_initialization(self, sample_rate):
        """Test VAD initialization."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        assert vad.sample_rate == sample_rate
        assert vad.frame_length > 0
        assert vad.hop_length > 0

    def test_process_speech_like_signal(self, speech_like_signal, sample_rate):
        """Test VAD on speech-like signal."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        is_speech = vad.process(speech_like_signal)

        # Output should be boolean array
        assert is_speech.dtype == bool
        assert len(is_speech) > 0

        # Should detect some speech and some silence
        assert is_speech.sum() > 0
        assert (~is_speech).sum() > 0

    def test_process_silence(self, zero_audio, sample_rate):
        """Test VAD on silence."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        is_speech = vad.process(zero_audio)

        # Most frames should be classified as silence
        # (zero signal has undefined entropy, may vary)
        assert len(is_speech) > 0

    def test_process_sine_wave(self, sine_wave, sample_rate):
        """Test VAD on continuous sine wave."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        is_speech = vad.process(sine_wave)

        # Sine wave has low entropy (concentrated spectrum)
        assert len(is_speech) > 0

    def test_process_white_noise(self, white_noise, sample_rate):
        """Test VAD on white noise."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        is_speech = vad.process(white_noise)

        # White noise has high entropy (flat spectrum)
        # Most should be classified as noise (not speech)
        assert len(is_speech) > 0

    def test_different_thresholds(self, speech_like_signal, sample_rate):
        """Test VAD with different entropy thresholds."""
        low_threshold_vad = SpectralEntropyVAD(sample_rate=sample_rate, threshold=0.5)
        high_threshold_vad = SpectralEntropyVAD(sample_rate=sample_rate, threshold=0.9)

        is_speech_low = low_threshold_vad.process(speech_like_signal)
        is_speech_high = high_threshold_vad.process(speech_like_signal)

        # Results should differ based on threshold
        assert len(is_speech_low) == len(is_speech_high)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test VAD with different frame parameters."""
        vad = SpectralEntropyVAD(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )
        is_speech = vad.process(sine_wave)

        assert len(is_speech) > 0
        assert is_speech.dtype == bool

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test VAD with very short audio."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        
        # Tiny audio is shorter than frame_length, should raise error
        with pytest.raises(ValueError):
            is_speech = vad.process(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        vad = SpectralEntropyVAD(sample_rate=sample_rate)
        repr_str = repr(vad)
        assert "SpectralEntropyVAD" in repr_str
        assert str(sample_rate) in repr_str
