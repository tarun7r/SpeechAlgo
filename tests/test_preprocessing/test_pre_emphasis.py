"""
Tests for speechalgo.preprocessing.pre_emphasis module.
"""

import numpy as np

from speechalgo.preprocessing.pre_emphasis import PreEmphasis


class TestPreEmphasis:
    """Tests for PreEmphasis class."""

    def test_initialization(self, sample_rate):
        """Test pre-emphasis initialization."""
        pre_emph = PreEmphasis()
        assert 0 < pre_emph.coefficient < 1

    def test_process_sine_wave(self, sine_wave, sample_rate):
        """Test pre-emphasis on sine wave."""
        pre_emph = PreEmphasis()
        emphasized = pre_emph.process(sine_wave)

        # Output should have same length
        assert emphasized.shape == sine_wave.shape
        assert emphasized.dtype == np.float32

    def test_emphasizes_high_frequencies(self, sample_rate):
        """Test that pre-emphasis boosts high frequencies."""
        # Create signal with low and high frequency components
        t = np.arange(sample_rate) / sample_rate
        low_freq = np.sin(2 * np.pi * 100 * t)
        high_freq = np.sin(2 * np.pi * 2000 * t)
        signal = (low_freq + high_freq).astype(np.float32)

        pre_emph = PreEmphasis(coefficient=0.97)
        emphasized = pre_emph.process(signal)

        # High frequency component should be relatively boosted
        # (difference between emphasized and original should be significant)
        assert not np.allclose(emphasized, signal)

    def test_inverse(self, sine_wave, sample_rate):
        """Test inverse pre-emphasis (de-emphasis)."""
        pre_emph = PreEmphasis(coefficient=0.97)

        # Apply pre-emphasis then inverse
        emphasized = pre_emph.process(sine_wave)
        restored = pre_emph.inverse(emphasized)

        # Should approximately restore original signal
        # (allow small numerical errors)
        np.testing.assert_allclose(restored, sine_wave, rtol=0.01, atol=0.01)

    def test_different_coefficients(self, sine_wave, sample_rate):
        """Test pre-emphasis with different coefficients."""
        for coef in [0.95, 0.97, 0.99]:
            pre_emph = PreEmphasis(coefficient=coef)
            emphasized = pre_emph.process(sine_wave)

            assert emphasized.shape == sine_wave.shape

    def test_zero_coefficient(self, sine_wave, sample_rate):
        """Test pre-emphasis with zero coefficient (no effect)."""
        pre_emph = PreEmphasis(coefficient=0.0)
        emphasized = pre_emph.process(sine_wave)

        # With coef=0, output should equal input
        np.testing.assert_array_equal(emphasized, sine_wave)

    def test_silence(self, zero_audio, sample_rate):
        """Test pre-emphasis on silence."""
        pre_emph = PreEmphasis()
        emphasized = pre_emph.process(zero_audio)

        # Silence should remain silence
        np.testing.assert_allclose(emphasized, 0.0, atol=1e-6)

    def test_first_sample(self, sample_rate):
        """Test that first sample is preserved."""
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        pre_emph = PreEmphasis(coefficient=0.97)
        emphasized = pre_emph.process(audio)

        # First sample should be unchanged
        assert emphasized[0] == audio[0]

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test pre-emphasis with very short audio."""
        pre_emph = PreEmphasis()
        emphasized = pre_emph.process(tiny_audio)

        assert emphasized.shape == tiny_audio.shape

    def test_single_sample(self, single_sample, sample_rate):
        """Test pre-emphasis with single sample."""
        pre_emph = PreEmphasis()
        emphasized = pre_emph.process(single_sample)

        # Single sample should be unchanged
        assert emphasized[0] == single_sample[0]

    def test_deterministic(self, sine_wave, sample_rate):
        """Test pre-emphasis is deterministic."""
        pre_emph = PreEmphasis()
        emphasized1 = pre_emph.process(sine_wave)
        emphasized2 = pre_emph.process(sine_wave)

        np.testing.assert_array_equal(emphasized1, emphasized2)

    def test_output_dtype(self, sine_wave, sample_rate):
        """Test pre-emphasis output is float32."""
        pre_emph = PreEmphasis()
        emphasized = pre_emph.process(sine_wave)

        assert emphasized.dtype == np.float32

    def test_repr(self, sample_rate):
        """Test string representation."""
        pre_emph = PreEmphasis()
        repr_str = repr(pre_emph)
        assert "PreEmphasis" in repr_str
