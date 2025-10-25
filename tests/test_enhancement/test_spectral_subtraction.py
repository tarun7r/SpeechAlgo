"""
Tests for speechalgo.enhancement.spectral_subtraction module.
"""

import numpy as np
import pytest

from speechalgo.enhancement.spectral_subtraction import SpectralSubtraction


class TestSpectralSubtraction:
    """Tests for SpectralSubtraction class."""

    def test_initialization(self, sample_rate):
        """Test spectral subtraction initialization."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        assert ss.sample_rate == sample_rate
        assert ss.oversubtraction_factor > 0
        assert 0 < ss.spectral_floor < 1

    def test_enhance_noisy_signal(self, noisy_sine, sample_rate, white_noise):
        """Test enhancement on noisy signal."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        # Estimate noise from a noise-only segment
        noise_segment = 0.1 * white_noise[:2000]
        ss.estimate_noise(noise_segment)
        enhanced = ss.process(noisy_sine)

        # Check output shape matches input
        assert enhanced.shape == noisy_sine.shape
        assert enhanced.dtype == np.float32

    def test_enhance_clean_signal(self, sine_wave, sample_rate, zero_audio):
        """Test enhancement on clean signal."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        # Estimate noise from silence (essentially no noise)
        ss.estimate_noise(zero_audio[:2000])
        enhanced = ss.process(sine_wave)

        # Should preserve clean signal reasonably well
        assert enhanced.shape == sine_wave.shape
        correlation = np.corrcoef(sine_wave, enhanced)[0, 1]
        assert correlation > 0.8

    def test_noise_estimation(self, noisy_sine, sample_rate, white_noise):
        """Test noise estimation from signal."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        # Estimate noise from a noise-only segment
        noise_segment = 0.1 * white_noise[:2000]
        ss.estimate_noise(noise_segment)
        # Process signal
        enhanced = ss.process(noisy_sine)

        assert enhanced.shape == noisy_sine.shape

    def test_different_parameters(self, noisy_sine, sample_rate, white_noise):
        """Test with different enhancement parameters."""
        noise_segment = 0.1 * white_noise[:2000]
        
        # Conservative enhancement
        ss_conservative = SpectralSubtraction(
            sample_rate=sample_rate,
            oversubtraction_factor=1.0,
            spectral_floor=0.1,
        )
        ss_conservative.estimate_noise(noise_segment)
        enhanced_conservative = ss_conservative.process(noisy_sine)

        # Aggressive enhancement
        ss_aggressive = SpectralSubtraction(
            sample_rate=sample_rate,
            oversubtraction_factor=3.0,
            spectral_floor=0.01,
        )
        ss_aggressive.estimate_noise(noise_segment)
        enhanced_aggressive = ss_aggressive.process(noisy_sine)

        assert enhanced_conservative.shape == enhanced_aggressive.shape

    def test_frame_parameters(self, noisy_sine, sample_rate, white_noise):
        """Test with different frame parameters."""
        ss = SpectralSubtraction(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )
        noise_segment = 0.1 * white_noise[:2000]
        ss.estimate_noise(noise_segment)
        enhanced = ss.process(noisy_sine)

        assert enhanced.shape == noisy_sine.shape

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test with very short audio."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        # Tiny audio (4 samples) is too short for processing, expect it to raise ValueError
        with pytest.raises(ValueError):
            enhanced = ss.process(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        ss = SpectralSubtraction(sample_rate=sample_rate)
        repr_str = repr(ss)
        assert "SpectralSubtraction" in repr_str
        assert str(sample_rate) in repr_str
