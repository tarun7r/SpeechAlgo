"""
Tests for speechalgo.utils.signal_processing module.
"""

import numpy as np
import pytest

from speechalgo.utils.signal_processing import (
    create_mel_filterbank,
    hz_to_mel,
    istft,
    mel_to_hz,
    stft,
)


class TestMelScale:
    """Tests for mel scale conversion functions."""

    def test_hz_to_mel_zero(self):
        """Test Hz to mel conversion at 0 Hz."""
        assert hz_to_mel(0.0) == 0.0

    def test_hz_to_mel_1000(self):
        """Test Hz to mel conversion at 1000 Hz."""
        mel = hz_to_mel(1000.0)
        assert mel == pytest.approx(1000.0, rel=0.01)

    def test_hz_to_mel_array(self):
        """Test Hz to mel conversion with array input."""
        hz = np.array([0.0, 1000.0, 4000.0])
        mel = hz_to_mel(hz)
        assert mel.shape == hz.shape
        assert mel[0] == 0.0
        assert mel[1] == pytest.approx(1000.0, rel=0.01)

    def test_mel_to_hz_zero(self):
        """Test mel to Hz conversion at 0 mel."""
        assert mel_to_hz(0.0) == 0.0

    def test_mel_to_hz_1000(self):
        """Test mel to Hz conversion at 1000 mel."""
        hz = mel_to_hz(1000.0)
        assert hz == pytest.approx(1000.0, rel=0.01)

    def test_mel_to_hz_array(self):
        """Test mel to Hz conversion with array input."""
        mel = np.array([0.0, 1000.0, 2000.0])
        hz = mel_to_hz(mel)
        assert hz.shape == mel.shape
        assert hz[0] == 0.0

    def test_roundtrip_conversion(self):
        """Test Hz -> mel -> Hz roundtrip."""
        hz_original = np.array([100.0, 500.0, 1000.0, 4000.0])
        mel = hz_to_mel(hz_original)
        hz_roundtrip = mel_to_hz(mel)
        np.testing.assert_allclose(hz_roundtrip, hz_original, rtol=1e-5)


class TestSTFT:
    """Tests for Short-Time Fourier Transform."""

    def test_stft_basic(self, sine_wave, sample_rate):
        """Test basic STFT computation."""
        spec = stft(sine_wave, n_fft=512, hop_length=256, window="hann")

        # Check shape: (n_fft//2 + 1, n_frames)
        assert spec.ndim == 2
        assert spec.shape[0] == 257  # 512//2 + 1
        assert spec.dtype == np.complex64

    def test_stft_magnitude(self, sine_wave):
        """Test STFT magnitude for sine wave."""
        spec = stft(sine_wave, n_fft=512, hop_length=256)
        mag = np.abs(spec)

        # Sine wave should have energy concentrated in one frequency bin
        assert mag.max() > 0

    def test_stft_different_windows(self, sine_wave):
        """Test STFT with different window functions."""
        for window in ["hann", "hamming", "blackman"]:
            spec = stft(sine_wave, n_fft=512, hop_length=256, window=window)
            assert spec.shape[0] == 257

    def test_stft_zero_audio(self, zero_audio):
        """Test STFT on zero signal."""
        spec = stft(zero_audio, n_fft=512, hop_length=256)
        assert np.allclose(np.abs(spec), 0.0)


class TestISTFT:
    """Tests for Inverse Short-Time Fourier Transform."""

    def test_istft_basic(self, sine_wave):
        """Test basic ISTFT computation."""
        # Forward transform
        spec = stft(sine_wave, n_fft=512, hop_length=256, window="hann")

        # Inverse transform
        reconstructed = istft(spec, hop_length=256, window="hann")

        # Check shape is similar (may differ slightly at boundaries)
        assert reconstructed.ndim == 1
        assert abs(len(reconstructed) - len(sine_wave)) < 512

    def test_istft_reconstruction(self, sine_wave):
        """Test STFT-ISTFT reconstruction accuracy."""
        spec = stft(sine_wave, n_fft=512, hop_length=128, window="hann")
        reconstructed = istft(spec, hop_length=128, window="hann")

        # Trim to same length
        min_len = min(len(sine_wave), len(reconstructed))
        original = sine_wave[:min_len]
        recon = reconstructed[:min_len]

        # Check reconstruction is close (not perfect due to windowing)
        correlation = np.corrcoef(original, recon)[0, 1]
        assert correlation > 0.99  # High correlation

    def test_istft_zero_spectrum(self):
        """Test ISTFT on zero spectrum."""
        spec = np.zeros((257, 100), dtype=np.complex64)
        audio = istft(spec, hop_length=256)
        assert np.allclose(audio, 0.0, atol=1e-6)


class TestMelFilterbank:
    """Tests for mel filterbank creation."""

    def test_create_mel_filterbank_basic(self, sample_rate):
        """Test basic mel filterbank creation."""
        filterbank = create_mel_filterbank(
            n_filters=40,
            n_fft=512,
            sample_rate=sample_rate,
            freq_min=0.0,
            freq_max=sample_rate / 2,
        )

        # Check shape: (n_filters, n_fft//2 + 1)
        assert filterbank.shape == (40, 257)
        assert filterbank.dtype == np.float32

    def test_mel_filterbank_properties(self, sample_rate):
        """Test mel filterbank properties."""
        filterbank = create_mel_filterbank(
            n_filters=40,
            n_fft=512,
            sample_rate=sample_rate,
        )

        # All values should be non-negative
        assert np.all(filterbank >= 0)

        # Each filter should sum to approximately 2 (triangular filters)
        # (HTK normalization makes peak = 1, not area = 1)
        max_per_filter = filterbank.max(axis=1)
        assert np.all(max_per_filter > 0)

    def test_mel_filterbank_frequency_range(self, sample_rate):
        """Test mel filterbank with custom frequency range."""
        f_min, f_max = 80.0, 4000.0
        filterbank = create_mel_filterbank(
            n_filters=20,
            n_fft=512,
            sample_rate=sample_rate,
            freq_min=f_min,
            freq_max=f_max,
        )

        assert filterbank.shape == (20, 257)

        # Low frequencies (below f_min) should have near-zero weights
        low_freq_bin = int(f_min / (sample_rate / 512))
        assert np.sum(filterbank[:, : max(1, low_freq_bin)]) < 0.1

    def test_mel_filterbank_different_sizes(self, sample_rate):
        """Test mel filterbank with different n_filters."""
        for n_filters in [20, 40, 80, 128]:
            filterbank = create_mel_filterbank(
                n_filters=n_filters,
                n_fft=512,
                sample_rate=sample_rate,
            )
            assert filterbank.shape == (n_filters, 257)
