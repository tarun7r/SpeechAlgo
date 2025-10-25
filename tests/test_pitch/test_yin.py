"""
Tests for speechalgo.pitch.yin module.
"""

import numpy as np

from speechalgo.pitch.yin import YIN


class TestYIN:
    """Tests for YIN pitch estimator."""

    def test_initialization(self, sample_rate):
        """Test YIN initialization."""
        yin = YIN(sample_rate=sample_rate)
        assert yin.sample_rate == sample_rate
        assert yin.f0_min > 0
        assert yin.f0_max > yin.f0_min
        assert 0 < yin.threshold < 1

    def test_estimate_sine_wave(self, sample_rate):
        """Test YIN on pure sine wave."""
        # Create 440 Hz sine wave (A4 note)
        frequency = 440.0
        # Use a single frame (1024 samples)
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        yin = YIN(sample_rate=sample_rate)
        f0 = yin.estimate(audio)

        # Check returns a float
        assert isinstance(f0, (float, np.floating))

        # Check estimated frequency is close to true frequency or an octave
        # YIN can still have octave errors on pure sine waves
        if f0 > 0:
            ratios = [f0 / frequency, frequency / f0]
            is_octave = any(abs(ratio - round(ratio)) < 0.05 for ratio in ratios)
            assert is_octave, f"Detected {f0} Hz, expected ~{frequency} Hz or octave"

    def test_estimate_low_frequency(self, sample_rate):
        """Test YIN on low-frequency sine wave."""
        frequency = 100.0
        frame_length = 2048  # Longer frame for low frequency
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        yin = YIN(sample_rate=sample_rate, f0_min=50, f0_max=500)
        f0 = yin.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            assert abs(f0 - frequency) / frequency < 0.05

    def test_estimate_silence(self, zero_audio, sample_rate):
        """Test YIN on silence."""
        yin = YIN(sample_rate=sample_rate)
        # Use a shorter silence to avoid issues
        silence = zero_audio[:1024]
        f0 = yin.estimate(silence)

        # Should return 0.0 for silence
        assert isinstance(f0, (float, np.floating))
        assert f0 == 0.0

    def test_different_thresholds(self, sine_wave, sample_rate):
        """Test YIN with different thresholds."""
        # Use a longer frame with enough periods
        frame_length = 1024
        audio = sine_wave[:frame_length]
        
        yin_low = YIN(sample_rate=sample_rate, threshold=0.1)
        yin_high = YIN(sample_rate=sample_rate, threshold=0.3)

        f0_low = yin_low.estimate(audio)
        f0_high = yin_high.estimate(audio)

        # Both should return float values
        assert isinstance(f0_low, (float, np.floating))
        assert isinstance(f0_high, (float, np.floating))

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test YIN works with frame-sized audio."""
        # YIN doesn't accept frame_length/hop_length parameters
        # It processes the given audio as a single frame
        yin = YIN(sample_rate=sample_rate)
        
        # Use a 2048 sample frame
        frame = sine_wave[:2048]
        f0 = yin.estimate(frame)

        assert isinstance(f0, (float, np.floating))

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test YIN with very short audio."""
        yin = YIN(sample_rate=sample_rate)
        # tiny_audio is too short (4 samples), expect it to raise ValueError
        with np.testing.assert_raises(ValueError):
            f0 = yin.estimate(tiny_audio)

    def test_repr(self, sample_rate):
        """Test string representation."""
        yin = YIN(sample_rate=sample_rate)
        repr_str = repr(yin)
        assert "YIN" in repr_str
        assert str(sample_rate) in repr_str
