"""
Tests for speechalgo.features.spectral module.
"""

import numpy as np
import pytest

from speechalgo.features.spectral import SpectralFeatures


class TestSpectralFeatures:
    """Tests for SpectralFeatures class."""

    def test_initialization(self, sample_rate):
        """Test spectral features initialization."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        assert spectral.sample_rate == sample_rate
        assert spectral.frame_length > 0
        assert spectral.hop_length > 0

    def test_spectral_centroid(self, sine_wave, sample_rate):
        """Test spectral centroid computation."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        centroid = spectral.spectral_centroid(sine_wave)

        # Check output
        assert centroid.ndim == 1
        assert len(centroid) > 0
        assert centroid.dtype == np.float32

        # Centroid should be in valid frequency range [0, sample_rate/2]
        assert np.all(centroid >= 0)
        assert np.all(centroid <= sample_rate / 2)

    def test_centroid_sine_wave_frequency(self, sample_rate):
        """Test centroid matches sine wave frequency."""
        # Create 440 Hz sine wave
        frequency = 440.0
        t = np.arange(sample_rate) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        spectral = SpectralFeatures(sample_rate=sample_rate)
        centroid = spectral.spectral_centroid(audio)

        # Centroid should be close to 440 Hz for pure sine wave
        mean_centroid = np.mean(centroid[centroid > 0])
        assert abs(mean_centroid - frequency) / frequency < 0.1

    def test_spectral_rolloff(self, sine_wave, sample_rate):
        """Test spectral rolloff computation."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        rolloff = spectral.spectral_rolloff(sine_wave)

        # Check output
        assert rolloff.ndim == 1
        assert len(rolloff) > 0
        assert rolloff.dtype == np.float32

        # Rolloff should be in valid frequency range
        assert np.all(rolloff >= 0)
        assert np.all(rolloff <= sample_rate / 2)

    def test_rolloff_different_percentiles(self, sine_wave, sample_rate):
        """Test rolloff with different percentiles."""
        spectral = SpectralFeatures(sample_rate=sample_rate)

        rolloff_85 = spectral.spectral_rolloff(sine_wave, percentile=0.85)
        rolloff_95 = spectral.spectral_rolloff(sine_wave, percentile=0.95)

        # Higher percentile should give higher frequency
        assert np.mean(rolloff_95) >= np.mean(rolloff_85)

    def test_spectral_flux(self, sine_wave, sample_rate):
        """Test spectral flux computation."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        flux = spectral.spectral_flux(sine_wave)

        # Check output
        assert flux.ndim == 1
        assert len(flux) > 0
        assert flux.dtype == np.float32

        # Flux should be non-negative
        assert np.all(flux >= 0)

    def test_flux_stable_vs_changing(self, sample_rate):
        """Test flux distinguishes stable from changing signals."""
        # Stable sine wave
        t = np.arange(sample_rate) / sample_rate
        stable = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Chirp (changing frequency)
        f0, f1 = 100.0, 1000.0
        changing = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2)).astype(np.float32)

        spectral = SpectralFeatures(sample_rate=sample_rate)
        flux_stable = spectral.spectral_flux(stable)
        flux_changing = spectral.spectral_flux(changing)

        # Changing signal should have higher flux
        assert np.mean(flux_changing) > np.mean(flux_stable)

    def test_spectral_bandwidth(self, sine_wave, sample_rate):
        """Test spectral bandwidth computation."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        bandwidth = spectral.spectral_bandwidth(sine_wave)

        # Check output
        assert bandwidth.ndim == 1
        assert len(bandwidth) > 0
        assert bandwidth.dtype == np.float32

        # Bandwidth should be non-negative
        assert np.all(bandwidth >= 0)

    def test_bandwidth_narrow_vs_wide(self, sample_rate):
        """Test bandwidth distinguishes narrow from wide spectra."""
        t = np.arange(sample_rate) / sample_rate

        # Narrow spectrum (pure sine)
        narrow = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Wide spectrum (white noise)
        wide = np.random.randn(sample_rate).astype(np.float32)

        spectral = SpectralFeatures(sample_rate=sample_rate)
        bw_narrow = spectral.spectral_bandwidth(narrow)
        bw_wide = spectral.spectral_bandwidth(wide)

        # Noise should have wider bandwidth
        assert np.mean(bw_wide) > np.mean(bw_narrow)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test with different frame parameters."""
        spectral = SpectralFeatures(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )

        centroid = spectral.spectral_centroid(sine_wave)
        rolloff = spectral.spectral_rolloff(sine_wave)
        flux = spectral.spectral_flux(sine_wave)
        bandwidth = spectral.spectral_bandwidth(sine_wave)

        # All features should have same length
        # (flux has first frame as zero since there's no previous frame)
        assert len(centroid) == len(rolloff) == len(bandwidth) == len(flux)

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test with very short audio."""
        spectral = SpectralFeatures(sample_rate=sample_rate)

        # tiny_audio is too short for STFT, expect it to raise ValueError
        with pytest.raises(ValueError):
            centroid = spectral.spectral_centroid(tiny_audio)
        with pytest.raises(ValueError):
            rolloff = spectral.spectral_rolloff(tiny_audio)
        with pytest.raises(ValueError):
            flux = spectral.spectral_flux(tiny_audio)
        with pytest.raises(ValueError):
            bandwidth = spectral.spectral_bandwidth(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        spectral = SpectralFeatures(sample_rate=sample_rate)
        repr_str = repr(spectral)
        assert "SpectralFeatures" in repr_str
        assert str(sample_rate) in repr_str
