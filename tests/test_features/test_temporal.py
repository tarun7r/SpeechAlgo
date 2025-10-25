"""
Tests for speechalgo.features.temporal module.
"""

import numpy as np
import pytest

from speechalgo.features.temporal import TemporalFeatures


class TestTemporalFeatures:
    """Tests for TemporalFeatures class."""

    def test_initialization(self, sample_rate):
        """Test temporal features initialization."""
        temporal = TemporalFeatures(sample_rate=sample_rate)
        assert temporal.sample_rate == sample_rate
        assert temporal.frame_length > 0
        assert temporal.hop_length > 0

    def test_zero_crossing_rate(self, sine_wave, sample_rate):
        """Test ZCR computation on sine wave."""
        temporal = TemporalFeatures(sample_rate=sample_rate)
        zcr = temporal.zero_crossing_rate(sine_wave)

        # Check output
        assert zcr.ndim == 1
        assert len(zcr) > 0
        assert zcr.dtype == np.float32

        # ZCR should be in valid range [0, 1]
        assert np.all(zcr >= 0)
        assert np.all(zcr <= 1)

    def test_zcr_voiced_vs_unvoiced(self, sample_rate):
        """Test ZCR distinguishes voiced from unvoiced."""
        temporal = TemporalFeatures(sample_rate=sample_rate)

        # Low frequency (voiced-like)
        t = np.arange(8000) / sample_rate
        voiced = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        zcr_voiced = temporal.zero_crossing_rate(voiced)

        # High frequency noise (unvoiced-like)
        unvoiced = np.random.randn(8000).astype(np.float32)
        zcr_unvoiced = temporal.zero_crossing_rate(unvoiced)

        # Unvoiced should have higher ZCR
        assert np.mean(zcr_unvoiced) > np.mean(zcr_voiced)

    def test_short_time_energy(self, sine_wave, sample_rate):
        """Test short-time energy computation."""
        temporal = TemporalFeatures(sample_rate=sample_rate)
        energy = temporal.short_time_energy(sine_wave)

        # Check output
        assert energy.ndim == 1
        assert len(energy) > 0
        assert energy.dtype == np.float32

        # Energy should be non-negative
        assert np.all(energy >= 0)

    def test_energy_silence_vs_signal(self, sine_wave, zero_audio, sample_rate):
        """Test energy distinguishes silence from signal."""
        temporal = TemporalFeatures(sample_rate=sample_rate)

        energy_signal = temporal.short_time_energy(sine_wave)
        energy_silence = temporal.short_time_energy(zero_audio)

        # Signal should have higher energy than silence
        assert np.mean(energy_signal) > np.mean(energy_silence)

    def test_root_mean_square(self, sine_wave, sample_rate):
        """Test RMS computation."""
        temporal = TemporalFeatures(sample_rate=sample_rate)
        rms = temporal.root_mean_square(sine_wave)

        # Check output
        assert rms.ndim == 1
        assert len(rms) > 0
        assert rms.dtype == np.float32

        # RMS should be non-negative
        assert np.all(rms >= 0)

    def test_rms_relationship_to_energy(self, sine_wave, sample_rate):
        """Test RMS is related to energy by sqrt."""
        temporal = TemporalFeatures(sample_rate=sample_rate)

        energy = temporal.short_time_energy(sine_wave)
        rms = temporal.root_mean_square(sine_wave)

        # RMS should be roughly sqrt(energy / frame_length)
        expected_rms = np.sqrt(energy / temporal.frame_length)
        np.testing.assert_allclose(rms, expected_rms, rtol=0.01)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test with different frame parameters."""
        temporal = TemporalFeatures(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )

        zcr = temporal.zero_crossing_rate(sine_wave)
        energy = temporal.short_time_energy(sine_wave)
        rms = temporal.root_mean_square(sine_wave)

        # All features should have same length
        assert len(zcr) == len(energy) == len(rms)

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test with very short audio."""
        temporal = TemporalFeatures(sample_rate=sample_rate)

        # tiny_audio is too short for frame-based processing, expect it to raise ValueError
        with pytest.raises(ValueError):
            zcr = temporal.zero_crossing_rate(tiny_audio)
        with pytest.raises(ValueError):
            energy = temporal.short_time_energy(tiny_audio)
        with pytest.raises(ValueError):
            rms = temporal.root_mean_square(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        temporal = TemporalFeatures(sample_rate=sample_rate)
        repr_str = repr(temporal)
        assert "TemporalFeatures" in repr_str
        assert str(sample_rate) in repr_str
