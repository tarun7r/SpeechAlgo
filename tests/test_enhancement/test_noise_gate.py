"""
Tests for speechalgo.enhancement.noise_gate module.
"""

import numpy as np
import pytest

from speechalgo.enhancement.noise_gate import NoiseGate


class TestNoiseGate:
    """Tests for NoiseGate class."""

    def test_initialization(self, sample_rate):
        """Test noise gate initialization."""
        gate = NoiseGate(sample_rate=sample_rate)
        assert gate.sample_rate == sample_rate
        assert gate.threshold_db < 0  # threshold in dB should be negative
        assert gate.attack_time > 0
        assert gate.release_time > 0

    def test_process_speech_like_signal(self, speech_like_signal, sample_rate):
        """Test noise gate on speech-like signal."""
        gate = NoiseGate(sample_rate=sample_rate, threshold_db=0.1)
        enhanced = gate.process(speech_like_signal)

        # Check output shape matches input
        assert enhanced.shape == speech_like_signal.shape
        assert enhanced.dtype == np.float32

        # Gate should attenuate quiet portions
        assert np.max(np.abs(enhanced)) <= np.max(np.abs(speech_like_signal))

    def test_process_silence(self, zero_audio, sample_rate):
        """Test noise gate on silence."""
        gate = NoiseGate(sample_rate=sample_rate)
        enhanced = gate.process(zero_audio)

        # Silence should remain silent
        assert np.allclose(enhanced, 0.0, atol=1e-6)

    def test_process_loud_signal(self, sine_wave, sample_rate):
        """Test noise gate on loud signal."""
        gate = NoiseGate(sample_rate=sample_rate, threshold_db=0.1)
        enhanced = gate.process(sine_wave)

        # Loud signal should pass through mostly unchanged
        correlation = np.corrcoef(sine_wave, enhanced)[0, 1]
        assert correlation > 0.95

    def test_different_thresholds(self, speech_like_signal, sample_rate):
        """Test with different gate thresholds."""
        low_gate = NoiseGate(sample_rate=sample_rate, threshold_db=0.05)
        high_gate = NoiseGate(sample_rate=sample_rate, threshold_db=0.2)

        enhanced_low = low_gate.process(speech_like_signal)
        enhanced_high = high_gate.process(speech_like_signal)

        # Higher threshold should attenuate more
        assert np.mean(np.abs(enhanced_high)) <= np.mean(np.abs(enhanced_low))

    def test_attack_release(self, sample_rate):
        """Test attack and release times."""
        # Create signal with sudden changes
        signal = np.concatenate(
            [
                np.zeros(1000, dtype=np.float32),
                np.ones(2000, dtype=np.float32),
                np.zeros(1000, dtype=np.float32),
            ]
        )

        gate = NoiseGate(
            sample_rate=sample_rate,
            threshold_db=0.5,
            attack_time=0.01,
            release_time=0.05,
        )
        enhanced = gate.process(signal)

        assert enhanced.shape == signal.shape

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test with different frame parameters."""
        gate = NoiseGate(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )
        enhanced = gate.process(sine_wave)

        assert enhanced.shape == sine_wave.shape

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test with very short audio."""
        gate = NoiseGate(sample_rate=sample_rate)
        
        # Tiny audio should still process but might raise error if too short
        with pytest.raises(ValueError):
            enhanced = gate.process(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        gate = NoiseGate(sample_rate=sample_rate)
        repr_str = repr(gate)
        assert "NoiseGate" in repr_str
        assert str(sample_rate) in repr_str
