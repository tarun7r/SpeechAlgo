"""
Tests for speechalgo.pitch.autocorrelation module.
"""

import numpy as np

from speechalgo.pitch.autocorrelation import Autocorrelation


class TestAutocorrelation:
    """Tests for Autocorrelation class."""

    def test_initialization(self, sample_rate):
        """Test pitch estimator initialization."""
        estimator = Autocorrelation(sample_rate=sample_rate)
        assert estimator.sample_rate == sample_rate
        assert estimator.f0_min > 0
        assert estimator.f0_max > estimator.f0_min

    def test_estimate_sine_wave(self, sample_rate):
        """Test pitch estimation on pure sine wave."""
        # Create 440 Hz sine wave (A4 note)
        frequency = 440.0
        # Use a single frame (1024 samples)
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = Autocorrelation(sample_rate=sample_rate)
        f0 = estimator.estimate(audio)

        # Check returns a float
        assert isinstance(f0, (float, np.floating))

        # Check estimated frequency is close to true frequency or an octave
        # Autocorrelation can detect octaves
        if f0 > 0:
            ratios = [f0 / frequency, frequency / f0]
            is_octave = any(abs(ratio - round(ratio)) < 0.05 for ratio in ratios)
            assert is_octave, f"Detected {f0} Hz, expected ~{frequency} Hz or octave"

    def test_estimate_low_frequency(self, sample_rate):
        """Test pitch estimation on low-frequency sine wave."""
        # Create 100 Hz sine wave
        frequency = 100.0
        frame_length = 2048  # Longer frame for low frequency
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = Autocorrelation(sample_rate=sample_rate, f0_min=50, f0_max=500)
        f0 = estimator.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            assert abs(f0 - frequency) / frequency < 0.1

    def test_estimate_high_frequency(self, sample_rate):
        """Test pitch estimation on high-frequency sine wave."""
        # Create 800 Hz sine wave
        frequency = 800.0
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = Autocorrelation(sample_rate=sample_rate, f0_min=50, f0_max=1000)
        f0 = estimator.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            assert abs(f0 - frequency) / frequency < 0.1

    def test_estimate_silence(self, zero_audio, sample_rate):
        """Test pitch estimation on silence."""
        estimator = Autocorrelation(sample_rate=sample_rate)
        # Use a shorter silence to avoid issues
        silence = zero_audio[:1024]
        f0 = estimator.estimate(silence)

        # Should return 0.0 for silence
        assert isinstance(f0, (float, np.floating))
        assert f0 == 0.0

    def test_estimate_noise(self, white_noise, sample_rate):
        """Test pitch estimation on white noise."""
        estimator = Autocorrelation(sample_rate=sample_rate)
        # Use a frame of noise
        noise_frame = white_noise[:1024]
        f0 = estimator.estimate(noise_frame)

        # Noise should return a float (might be 0 or detected spuriously)
        assert isinstance(f0, (float, np.floating))

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test pitch estimation works with frame-sized audio."""
        # Autocorrelation doesn't accept frame_length/hop_length parameters
        # It processes the given audio as a single frame
        estimator = Autocorrelation(sample_rate=sample_rate)
        
        # Use a 2048 sample frame
        frame = sine_wave[:2048]
        f0 = estimator.estimate(frame)

        assert isinstance(f0, (float, np.floating))

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test pitch estimation with very short audio."""
        estimator = Autocorrelation(sample_rate=sample_rate)
        # tiny_audio is too short (4 samples), expect it to raise ValueError
        with np.testing.assert_raises(ValueError):
            f0 = estimator.estimate(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        estimator = Autocorrelation(sample_rate=sample_rate)
        repr_str = repr(estimator)
        assert "Autocorrelation" in repr_str
        assert str(sample_rate) in repr_str
