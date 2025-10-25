"""
Tests for speechalgo.vad.energy_based module.
"""

import numpy as np
import pytest

from speechalgo.vad.energy_based import EnergyBasedVAD


class TestEnergyBasedVAD:
    """Tests for EnergyBasedVAD class."""

    def test_initialization(self, sample_rate):
        """Test VAD initialization."""
        vad = EnergyBasedVAD(sample_rate=sample_rate)
        assert vad.sample_rate == sample_rate
        assert vad.frame_length > 0
        assert vad.hop_length > 0

    def test_process_speech_like_signal(self, speech_like_signal, sample_rate):
        """Test VAD on speech-like signal."""
        vad = EnergyBasedVAD(sample_rate=sample_rate, threshold=0.05)
        is_speech = vad.process(speech_like_signal)

        # Output should be boolean array
        assert is_speech.dtype == bool
        assert len(is_speech) > 0

        # Should detect some speech (at least some frames active)
        assert is_speech.sum() > 0

    def test_process_silence(self, zero_audio, sample_rate):
        """Test VAD on silence."""
        vad = EnergyBasedVAD(sample_rate=sample_rate, threshold=0.01)
        is_speech = vad.process(zero_audio)

        # All frames should be classified as silence
        assert is_speech.sum() == 0

    def test_process_sine_wave(self, sine_wave, sample_rate):
        """Test VAD on continuous sine wave."""
        vad = EnergyBasedVAD(sample_rate=sample_rate, threshold=0.01)
        is_speech = vad.process(sine_wave)

        # Most frames should be classified as speech
        speech_ratio = is_speech.sum() / len(is_speech)
        assert speech_ratio > 0.8

    def test_different_thresholds(self, noisy_sine, sample_rate):
        """Test VAD with different energy thresholds."""
        low_threshold_vad = EnergyBasedVAD(sample_rate=sample_rate, threshold=0.001)
        high_threshold_vad = EnergyBasedVAD(sample_rate=sample_rate, threshold=0.1)

        is_speech_low = low_threshold_vad.process(noisy_sine)
        is_speech_high = high_threshold_vad.process(noisy_sine)

        # Lower threshold should detect more speech
        assert is_speech_low.sum() >= is_speech_high.sum()

    def test_hangover(self, sample_rate):
        """Test hangover mechanism."""
        # Create signal with brief silence
        signal = np.concatenate(
            [
                np.ones(2000, dtype=np.float32),  # Speech
                np.zeros(200, dtype=np.float32),  # Brief silence
                np.ones(2000, dtype=np.float32),  # Speech
            ]
        )

        vad = EnergyBasedVAD(sample_rate=sample_rate, hangover=3)
        is_speech = vad.process(signal)

        # Hangover should bridge the gap
        assert is_speech.sum() > 0

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test VAD with different frame parameters."""
        vad = EnergyBasedVAD(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )
        is_speech = vad.process(sine_wave)

        assert len(is_speech) > 0
        assert is_speech.dtype == bool

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test VAD with very short audio."""
        vad = EnergyBasedVAD(sample_rate=sample_rate)
        
        # Tiny audio is shorter than frame_length, should raise error
        with pytest.raises(ValueError, match="length"):
            is_speech = vad.process(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        vad = EnergyBasedVAD(sample_rate=sample_rate)
        repr_str = repr(vad)
        assert "EnergyBasedVAD" in repr_str
        assert str(sample_rate) in repr_str
