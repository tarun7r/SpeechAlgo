"""
Tests for speechalgo.preprocessing.mel_spectrogram module.
"""

import numpy as np

from speechalgo.preprocessing.mel_spectrogram import MelSpectrogram


class TestMelSpectrogram:
    """Tests for MelSpectrogram class."""

    def test_initialization(self, sample_rate):
        """Test mel-spectrogram initialization."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        assert mel_spec.sample_rate == sample_rate
        assert mel_spec.n_mels > 0
        assert mel_spec.n_fft > 0

    def test_process_sine_wave(self, sine_wave, sample_rate):
        """Test mel-spectrogram on sine wave."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=40)
        spec = mel_spec.process(sine_wave)

        # Check shape: (n_mels, n_frames)
        assert spec.ndim == 2
        assert spec.shape[0] == 40
        assert spec.shape[1] > 0
        assert spec.dtype == np.float32

        # All values should be non-negative
        assert np.all(spec >= 0)

    def test_different_n_mels(self, sine_wave, sample_rate):
        """Test mel-spectrogram with different number of mel bands."""
        for n_mels in [20, 40, 80, 128]:
            mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
            spec = mel_spec.process(sine_wave)

            assert spec.shape[0] == n_mels

    def test_to_db(self, sine_wave, sample_rate):
        """Test conversion to decibels."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        spec = mel_spec.process(sine_wave)
        spec_db = mel_spec.to_db(spec)

        # Shape should be preserved
        assert spec_db.shape == spec.shape

        # dB values should be negative or zero (log scale)
        assert np.all(spec_db <= 0)

    def test_energy_concentration(self, sample_rate):
        """Test that sine wave energy is concentrated in one mel band."""
        # Create 440 Hz sine wave
        frequency = 440.0
        t = np.arange(sample_rate) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=40)
        spec = mel_spec.process(audio)

        # Energy should be concentrated (max should be much larger than mean)
        max_energy = np.max(spec)
        mean_energy = np.mean(spec)
        assert max_energy > 10 * mean_energy

    def test_silence(self, zero_audio, sample_rate):
        """Test mel-spectrogram on silence."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        spec = mel_spec.process(zero_audio)

        # Should produce near-zero spectrogram
        assert spec.shape[0] == 40
        assert np.all(spec < 1e-6)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test mel-spectrogram with different frame parameters."""
        mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
        )
        spec = mel_spec.process(sine_wave)

        assert spec.shape[0] == 40
        assert spec.shape[1] > 0

    def test_frequency_range(self, sine_wave, sample_rate):
        """Test mel-spectrogram with custom frequency range."""
        mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            freq_min=80.0,
            freq_max=4000.0,
        )
        spec = mel_spec.process(sine_wave)

        assert spec.shape[0] == 40

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test mel-spectrogram with very short audio."""
        # Tiny audio (4 samples) is too short, expect it to raise ValueError
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        with np.testing.assert_raises(ValueError):
            spec = mel_spec.process(tiny_audio)

    def test_deterministic(self, sine_wave, sample_rate):
        """Test mel-spectrogram is deterministic."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        spec1 = mel_spec.process(sine_wave)
        spec2 = mel_spec.process(sine_wave)

        np.testing.assert_array_equal(spec1, spec2)

    def test_db_ref_parameter(self, sine_wave, sample_rate):
        """Test to_db with different reference values."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        spec = mel_spec.process(sine_wave)

        spec_db1 = mel_spec.to_db(spec, ref=1.0)
        spec_db2 = mel_spec.to_db(spec, ref=np.max(spec))

        # Different references should give different results
        assert not np.allclose(spec_db1, spec_db2)

    def test_output_dtype(self, sine_wave, sample_rate):
        """Test mel-spectrogram output is float32."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        spec = mel_spec.process(sine_wave)

        assert spec.dtype == np.float32

    def test_repr(self, sample_rate):
        """Test string representation."""
        mel_spec = MelSpectrogram(sample_rate=sample_rate)
        repr_str = repr(mel_spec)
        assert "MelSpectrogram" in repr_str
        assert str(sample_rate) in repr_str
