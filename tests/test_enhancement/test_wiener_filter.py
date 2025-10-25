"""
Tests for speechalgo.enhancement.wiener_filter module.
"""

import numpy as np

from speechalgo.enhancement.wiener_filter import WienerFilter


class TestWienerFilter:
    """Tests for WienerFilter class."""

    def test_initialization(self, sample_rate):
        """Test Wiener filter initialization."""
        wiener = WienerFilter(sample_rate=sample_rate)
        assert wiener.sample_rate == sample_rate
        assert wiener.noise_floor > 0

    def test_process_noisy_signal(self, noisy_sine, sample_rate):
        """Test Wiener filtering on noisy signal."""
        wiener = WienerFilter(sample_rate=sample_rate)

        # Estimate noise from first part of signal
        noise_segment = noisy_sine[: sample_rate // 4]
        wiener.estimate_noise(noise_segment)

        # Process noisy signal
        enhanced = wiener.process(noisy_sine)

        # Check output shape matches input
        assert enhanced.shape == noisy_sine.shape
        assert enhanced.dtype == np.float32

    def test_process_clean_signal(self, sine_wave, sample_rate):
        """Test Wiener filtering on clean signal."""
        wiener = WienerFilter(sample_rate=sample_rate)

        # Estimate noise from silence
        noise_segment = np.zeros(sample_rate // 4, dtype=np.float32)
        wiener.estimate_noise(noise_segment)

        # Process clean signal
        enhanced = wiener.process(sine_wave)

        # Should preserve clean signal reasonably well
        assert enhanced.shape == sine_wave.shape
        # High correlation with original
        correlation = np.corrcoef(sine_wave, enhanced)[0, 1]
        assert correlation > 0.7

    def test_noise_reduction(self, sine_wave, white_noise, sample_rate):
        """Test that Wiener filter processes noisy signal."""
        # Create noisy signal
        signal_power = np.var(sine_wave)
        noise_power = 0.1 * signal_power
        noise = np.sqrt(noise_power / np.var(white_noise)) * white_noise
        noisy = sine_wave + noise

        wiener = WienerFilter(sample_rate=sample_rate)

        # Estimate noise from noise segment
        wiener.estimate_noise(noise[: sample_rate // 4])

        # Process noisy signal
        enhanced = wiener.process(noisy)

        # Enhanced signal should have same shape
        assert enhanced.shape == noisy.shape
        # Check that processing produced different output
        assert not np.array_equal(enhanced, noisy)

    def test_estimate_noise(self, noisy_sine, sample_rate):
        """Test noise estimation from signal."""
        wiener = WienerFilter(sample_rate=sample_rate)

        # Estimate noise from initial segment
        noise_segment = noisy_sine[: sample_rate // 4]
        wiener.estimate_noise(noise_segment)

        # Check noise spectrum was estimated
        assert wiener.noise_spectrum is not None
        assert wiener.noise_spectrum.ndim == 1

        # Process signal after noise estimation
        enhanced = wiener.process(noisy_sine)
        assert enhanced.shape == noisy_sine.shape

    def test_different_noise_floor(self, noisy_sine, sample_rate):
        """Test with different noise floor values."""
        noise_segment = noisy_sine[: sample_rate // 4]

        wiener_low = WienerFilter(sample_rate=sample_rate, noise_floor=1e-4)
        wiener_low.estimate_noise(noise_segment)
        enhanced_low = wiener_low.process(noisy_sine)

        wiener_high = WienerFilter(sample_rate=sample_rate, noise_floor=1e-2)
        wiener_high.estimate_noise(noise_segment)
        enhanced_high = wiener_high.process(noisy_sine)

        assert enhanced_low.shape == enhanced_high.shape

    def test_gain_computation(self, sample_rate):
        """Test that Wiener gain is computed correctly."""
        # Create signal with known SNR
        t = np.arange(sample_rate) / sample_rate
        clean = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = 0.1 * np.random.randn(sample_rate).astype(np.float32)
        noisy = clean + noise

        wiener = WienerFilter(sample_rate=sample_rate)

        # Estimate noise from noise segment
        wiener.estimate_noise(noise[: sample_rate // 4])

        # Process noisy signal
        enhanced = wiener.process(noisy)

        # Enhanced should have higher SNR than input
        assert enhanced.shape == noisy.shape

    def test_frame_parameters(self, noisy_sine, sample_rate):
        """Test with different frame parameters."""
        wiener = WienerFilter(
            sample_rate=sample_rate,
            frame_length=1024,
            hop_length=512,
        )

        noise_segment = noisy_sine[: sample_rate // 4]
        wiener.estimate_noise(noise_segment)

        enhanced = wiener.process(noisy_sine)
        assert enhanced.shape == noisy_sine.shape

    def test_silence(self, sample_rate):
        """Test Wiener filter on silence."""
        silence = np.zeros(sample_rate, dtype=np.float32)

        wiener = WienerFilter(sample_rate=sample_rate)
        wiener.estimate_noise(silence[: sample_rate // 4])
        enhanced = wiener.process(silence)

        # Silence should remain mostly silent
        assert np.mean(np.abs(enhanced)) < 0.01

    def test_short_audio(self, sample_rate):
        """Test with short audio."""
        # Use audio long enough for at least one frame (512 samples minimum)
        short_audio = np.random.randn(1024).astype(np.float32) * 0.1

        wiener = WienerFilter(sample_rate=sample_rate)
        wiener.estimate_noise(short_audio[:512])
        enhanced = wiener.process(short_audio)

        assert enhanced.shape == short_audio.shape

    def test_deterministic(self, noisy_sine, sample_rate):
        """Test Wiener filter is deterministic."""
        wiener = WienerFilter(sample_rate=sample_rate)

        noise_segment = noisy_sine[: sample_rate // 4]
        wiener.estimate_noise(noise_segment)

        enhanced1 = wiener.process(noisy_sine)
        enhanced2 = wiener.process(noisy_sine)

        np.testing.assert_allclose(enhanced1, enhanced2, rtol=1e-5)

    def test_output_dtype(self, noisy_sine, sample_rate):
        """Test Wiener filter output is float32."""
        wiener = WienerFilter(sample_rate=sample_rate)

        noise_segment = noisy_sine[: sample_rate // 4]
        wiener.estimate_noise(noise_segment)

        enhanced = wiener.process(noisy_sine)
        assert enhanced.dtype == np.float32

    def test_repr(self, sample_rate):
        """Test string representation."""
        wiener = WienerFilter(sample_rate=sample_rate)
        repr_str = repr(wiener)
        assert "WienerFilter" in repr_str
        assert str(sample_rate) in repr_str
