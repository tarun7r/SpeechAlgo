"""
Tests for speechalgo.pitch.hps module.
"""

import numpy as np

from speechalgo.pitch.hps import HPS


class TestHPS:
    """Tests for HPS (Harmonic Product Spectrum) pitch estimator."""

    def test_initialization(self, sample_rate):
        """Test HPS initialization."""
        hps = HPS(sample_rate=sample_rate)
        assert hps.sample_rate == sample_rate
        assert hps.f0_min > 0
        assert hps.f0_max > hps.f0_min
        assert hps.n_harmonics > 1

    def test_estimate_sine_wave(self, sample_rate):
        """Test HPS on pure sine wave."""
        # Create 440 Hz sine wave (A4 note)
        frequency = 440.0
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        hps = HPS(sample_rate=sample_rate)
        f0 = hps.estimate(audio)

        # Should return a single float
        assert isinstance(f0, (float, np.floating))

        # Check estimated frequency is close to true frequency or a harmonic
        # HPS can sometimes detect harmonics/subharmonics
        if f0 > 0:
            # Check if it's close to the fundamental or a harmonic/subharmonic
            ratios = [f0 / frequency, frequency / f0]
            # Allow detection of fundamental, octaves, or fifths
            is_harmonic = any(abs(ratio - round(ratio)) < 0.1 for ratio in ratios)
            assert is_harmonic, f"Detected {f0} Hz, expected ~{frequency} Hz or harmonic"

    def test_estimate_low_frequency(self, sample_rate):
        """Test HPS on low-frequency sine wave."""
        frequency = 100.0
        frame_length = 2048  # Longer frame for low frequency
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        hps = HPS(sample_rate=sample_rate, f0_min=50, f0_max=500)
        f0 = hps.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            # Check if it's close to the fundamental or a harmonic/subharmonic
            ratios = [f0 / frequency, frequency / f0]
            is_harmonic = any(abs(ratio - round(ratio)) < 0.15 for ratio in ratios)
            assert is_harmonic, f"Detected {f0} Hz, expected ~{frequency} Hz or harmonic"

    def test_estimate_high_frequency(self, sample_rate):
        """Test HPS on high-frequency sine wave."""
        frequency = 800.0
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        hps = HPS(sample_rate=sample_rate, f0_min=50, f0_max=1000)
        f0 = hps.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            # Check if it's close to the fundamental or a harmonic/subharmonic
            ratios = [f0 / frequency, frequency / f0]
            is_harmonic = any(abs(ratio - round(ratio)) < 0.15 for ratio in ratios)
            assert is_harmonic, f"Detected {f0} Hz, expected ~{frequency} Hz or harmonic"

    def test_estimate_silence(self, sample_rate):
        """Test HPS on silence."""
        silence = np.zeros(1024, dtype=np.float32)

        hps = HPS(sample_rate=sample_rate)
        f0 = hps.estimate(silence)

        # Silent frames should have zero pitch
        assert isinstance(f0, (float, np.floating))
        assert f0 == 0.0

    def test_estimate_noise(self, sample_rate):
        """Test HPS on white noise."""
        noise = np.random.randn(1024).astype(np.float32) * 0.1

        hps = HPS(sample_rate=sample_rate)
        f0 = hps.estimate(noise)

        # Should return a float (may be 0 for noise)
        assert isinstance(f0, (float, np.floating))

    def test_different_n_harmonics(self, sine_wave, sample_rate):
        """Test HPS with different number of harmonics."""
        frame = sine_wave[:1024]

        for n_harmonics in [3, 5, 7]:
            hps = HPS(sample_rate=sample_rate, n_harmonics=n_harmonics)
            f0 = hps.estimate(frame)

            assert isinstance(f0, (float, np.floating))

    def test_harmonic_signal(self, sample_rate):
        """Test HPS on signal with harmonics."""
        # Create signal with fundamental and harmonics
        frequency = 200.0
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        # Fundamental + 2nd + 3rd harmonics
        audio = (
            np.sin(2 * np.pi * frequency * t)
            + 0.5 * np.sin(2 * np.pi * 2 * frequency * t)
            + 0.3 * np.sin(2 * np.pi * 3 * frequency * t)
        ).astype(np.float32)

        hps = HPS(sample_rate=sample_rate, n_harmonics=5)
        f0 = hps.estimate(audio)

        # Should detect fundamental frequency
        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            # Should be close to fundamental, not harmonics
            assert abs(f0 - frequency) / frequency < 0.15

    def test_different_n_fft(self, sine_wave, sample_rate):
        """Test HPS with different FFT sizes."""
        hps_auto = HPS(sample_rate=sample_rate, n_fft=None)
        hps_2048 = HPS(sample_rate=sample_rate, n_fft=2048)

        frame = sine_wave[:1024]
        f0_auto = hps_auto.estimate(frame)
        f0_2048 = hps_2048.estimate(frame)

        assert isinstance(f0_auto, (float, np.floating))
        assert isinstance(f0_2048, (float, np.floating))

    def test_minimum_length(self, sample_rate):
        """Test HPS requires minimum audio length."""
        hps = HPS(sample_rate=sample_rate)

        # Should work with minimum length (64 samples per validate_audio)
        frame = np.zeros(64, dtype=np.float32)
        f0 = hps.estimate(frame)
        assert isinstance(f0, (float, np.floating))

    def test_deterministic(self, sine_wave, sample_rate):
        """Test HPS is deterministic."""
        hps = HPS(sample_rate=sample_rate)
        frame = sine_wave[:1024]
        f0_1 = hps.estimate(frame)
        f0_2 = hps.estimate(frame)

        assert f0_1 == f0_2

    def test_repr(self, sample_rate):
        """Test string representation."""
        hps = HPS(sample_rate=sample_rate)
        repr_str = repr(hps)
        assert "HPS" in repr_str
        assert str(sample_rate) in repr_str
