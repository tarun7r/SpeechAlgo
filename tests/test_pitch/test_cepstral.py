"""
Tests for speechalgo.pitch.cepstral module.
"""

import numpy as np

from speechalgo.pitch.cepstral import CepstralPitch


class TestCepstralPitch:
    """Tests for CepstralPitch class."""

    def test_initialization(self, sample_rate):
        """Test cepstral pitch estimator initialization."""
        estimator = CepstralPitch(sample_rate=sample_rate)
        assert estimator.sample_rate == sample_rate
        assert estimator.f0_min > 0
        assert estimator.f0_max > estimator.f0_min

    def test_estimate_sine_wave(self, sample_rate):
        """Test cepstral pitch estimation on pure sine wave."""
        # Create 440 Hz sine wave (A4 note)
        frequency = 440.0
        # Use a single frame (1024 samples = ~64ms at 16kHz)
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = CepstralPitch(sample_rate=sample_rate)
        f0 = estimator.estimate(audio)

        # Should return a single float
        assert isinstance(f0, (float, np.floating))

        # Check estimated frequency is close to true frequency or a harmonic
        # Cepstral method can detect harmonics/subharmonics
        if f0 > 0:
            ratios = [f0 / frequency, frequency / f0]
            is_harmonic = any(abs(ratio - round(ratio)) < 0.15 for ratio in ratios)
            assert is_harmonic, f"Detected {f0} Hz, expected ~{frequency} Hz or harmonic"

    def test_estimate_low_frequency(self, sample_rate):
        """Test cepstral pitch estimation on low-frequency sine wave."""
        frequency = 100.0
        frame_length = 2048  # Longer frame for low frequency
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = CepstralPitch(sample_rate=sample_rate, f0_min=50, f0_max=500)
        f0 = estimator.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            assert abs(f0 - frequency) / frequency < 0.2

    def test_estimate_high_frequency(self, sample_rate):
        """Test cepstral pitch estimation on high-frequency sine wave."""
        frequency = 800.0
        frame_length = 1024
        t = np.arange(frame_length) / sample_rate
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        estimator = CepstralPitch(sample_rate=sample_rate, f0_min=50, f0_max=1000)
        f0 = estimator.estimate(audio)

        assert isinstance(f0, (float, np.floating))
        if f0 > 0:
            ratios = [f0 / frequency, frequency / f0]
            is_harmonic = any(abs(ratio - round(ratio)) < 0.2 for ratio in ratios)
            assert is_harmonic, f"Detected {f0} Hz, expected ~{frequency} Hz or harmonic"

    def test_estimate_silence(self, sample_rate):
        """Test cepstral pitch estimation on silence."""
        # Create silent frame
        silence = np.zeros(1024, dtype=np.float32)

        estimator = CepstralPitch(sample_rate=sample_rate)
        f0 = estimator.estimate(silence)

        # Silent frames should have zero pitch
        assert isinstance(f0, (float, np.floating))
        assert f0 == 0.0

    def test_estimate_noise(self, sample_rate):
        """Test cepstral pitch estimation on white noise."""
        # Create noise frame
        noise = np.random.randn(1024).astype(np.float32) * 0.1

        estimator = CepstralPitch(sample_rate=sample_rate)
        f0 = estimator.estimate(noise)

        # Should return a float (may be 0 for noise)
        assert isinstance(f0, (float, np.floating))

    def test_lifter_parameter(self, sine_wave, sample_rate):
        """Test with different lifter cutoff values."""
        # Liftering can improve pitch estimation
        estimator_no_lift = CepstralPitch(sample_rate=sample_rate, lifter_cutoff=0)
        estimator_with_lift = CepstralPitch(sample_rate=sample_rate, lifter_cutoff=80)

        frame = sine_wave[:1024]
        f0_no_lift = estimator_no_lift.estimate(frame)
        f0_with_lift = estimator_with_lift.estimate(frame)

        # Both should return floats
        assert isinstance(f0_no_lift, (float, np.floating))
        assert isinstance(f0_with_lift, (float, np.floating))

    def test_different_n_fft(self, sine_wave, sample_rate):
        """Test cepstral pitch estimation with different FFT sizes."""
        estimator_auto = CepstralPitch(sample_rate=sample_rate, n_fft=None)
        estimator_2048 = CepstralPitch(sample_rate=sample_rate, n_fft=2048)

        frame = sine_wave[:1024]
        f0_auto = estimator_auto.estimate(frame)
        f0_2048 = estimator_2048.estimate(frame)

        assert isinstance(f0_auto, (float, np.floating))
        assert isinstance(f0_2048, (float, np.floating))

    def test_minimum_length(self, sample_rate):
        """Test cepstral pitch estimation requires minimum length."""
        estimator = CepstralPitch(sample_rate=sample_rate, f0_min=80)
        # quefrency_max = sample_rate / f0_min
        min_length = int(sample_rate / 80)

        # Frame at minimum length should work
        frame = np.zeros(min_length, dtype=np.float32)
        f0 = estimator.estimate(frame)
        assert isinstance(f0, (float, np.floating))

    def test_deterministic(self, sine_wave, sample_rate):
        """Test cepstral pitch estimation is deterministic."""
        estimator = CepstralPitch(sample_rate=sample_rate)
        frame = sine_wave[:1024]
        f0_1 = estimator.estimate(frame)
        f0_2 = estimator.estimate(frame)

        assert f0_1 == f0_2

    def test_repr(self, sample_rate):
        """Test string representation."""
        estimator = CepstralPitch(sample_rate=sample_rate)
        repr_str = repr(estimator)
        assert "CepstralPitch" in repr_str
        assert str(sample_rate) in repr_str
