"""
Tests for speechalgo.utils.validation module.
"""

import numpy as np
import pytest

from speechalgo.utils.validation import (
    validate_audio,
    validate_frame_length,
    validate_hop_length,
    validate_sample_rate,
)


class TestValidateAudio:
    """Tests for validate_audio function."""

    def test_valid_1d_audio(self, sine_wave):
        """Test validation passes for valid 1D audio."""
        validate_audio(sine_wave)  # Should not raise

    def test_valid_float32(self):
        """Test validation passes for float32 audio."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        validate_audio(audio)

    def test_valid_float64(self):
        """Test validation passes for float64 audio."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        validate_audio(audio)

    def test_empty_audio(self):
        """Test validation fails for empty audio."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="audio cannot be empty"):
            validate_audio(audio)

    def test_2d_audio(self):
        """Test validation fails for 2D audio."""
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(ValueError, match="audio has 2 channels"):
            validate_audio(audio)

    def test_non_float_audio(self):
        """Test validation passes for int audio (auto-converted to float)."""
        audio = np.array([1, 2, 3], dtype=np.int32)
        # Validation should pass since we don't check dtype
        validate_audio(audio)

    def test_none_audio(self):
        """Test validation fails for None."""
        with pytest.raises(TypeError):
            validate_audio(None)


class TestValidateSampleRate:
    """Tests for validate_sample_rate function."""

    def test_valid_sample_rates(self):
        """Test validation passes for valid sample rates."""
        for sr in [8000, 16000, 22050, 44100, 48000]:
            validate_sample_rate(sr)

    def test_zero_sample_rate(self):
        """Test validation fails for zero sample rate."""
        with pytest.raises(ValueError, match="sample_rate.*must be between"):
            validate_sample_rate(0)

    def test_negative_sample_rate(self):
        """Test validation fails for negative sample rate."""
        with pytest.raises(ValueError, match="sample_rate.*must be between"):
            validate_sample_rate(-16000)

    def test_non_integer_sample_rate(self):
        """Test validation fails for non-integer sample rate."""
        with pytest.raises(TypeError):
            validate_sample_rate(16000.5)


class TestValidateFrameLength:
    """Tests for validate_frame_length function."""

    def test_valid_frame_lengths(self):
        """Test validation passes for valid frame lengths."""
        for length in [128, 256, 512, 1024, 2048]:
            validate_frame_length(length)

    def test_zero_frame_length(self):
        """Test validation fails for zero frame length."""
        with pytest.raises(ValueError, match="frame_length must be positive"):
            validate_frame_length(0)

    def test_negative_frame_length(self):
        """Test validation fails for negative frame length."""
        with pytest.raises(ValueError, match="frame_length must be positive"):
            validate_frame_length(-512)


class TestValidateHopLength:
    """Tests for validate_hop_length function."""

    def test_valid_hop_lengths(self):
        """Test validation passes for valid hop lengths."""
        for length in [64, 128, 256, 512]:
            validate_hop_length(length)

    def test_zero_hop_length(self):
        """Test validation fails for zero hop length."""
        with pytest.raises(ValueError, match="hop_length must be positive"):
            validate_hop_length(0)

    def test_negative_hop_length(self):
        """Test validation fails for negative hop length."""
        with pytest.raises(ValueError, match="hop_length must be positive"):
            validate_hop_length(-256)

    def test_hop_greater_than_frame(self):
        """Test validation fails when hop length exceeds frame length."""
        with pytest.raises(ValueError, match="hop_length.*should not exceed frame_length"):
            validate_hop_length(1024, frame_length=512)
