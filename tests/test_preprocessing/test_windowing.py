"""
Tests for speechalgo.preprocessing.windowing module.
"""

import numpy as np
import pytest

from speechalgo.preprocessing.windowing import (
    blackman_window,
    hamming_window,
    hanning_window,
)


class TestHammingWindow:
    """Tests for Hamming window function."""

    def test_hamming_basic(self):
        """Test basic Hamming window generation."""
        window = hamming_window(512)
        assert len(window) == 512
        assert window.dtype == np.float32

    def test_hamming_symmetry(self):
        """Test Hamming window is symmetric."""
        window = hamming_window(512)
        np.testing.assert_allclose(window[:256], window[-256:][::-1], rtol=1e-5)

    def test_hamming_peak(self):
        """Test Hamming window has peak at center."""
        window = hamming_window(512)
        assert window[256] == pytest.approx(1.0, abs=0.01)

    def test_hamming_edges(self):
        """Test Hamming window has low values at edges."""
        window = hamming_window(512)
        assert window[0] < 0.1
        assert window[-1] < 0.1

    def test_hamming_different_sizes(self):
        """Test Hamming window with different sizes."""
        for size in [64, 128, 256, 512, 1024]:
            window = hamming_window(size)
            assert len(window) == size
            assert window.max() == pytest.approx(1.0, abs=0.01)

    def test_hamming_invalid_size(self):
        """Test Hamming window with invalid size."""
        with pytest.raises(ValueError):
            hamming_window(0)
        with pytest.raises(ValueError):
            hamming_window(-512)


class TestHanningWindow:
    """Tests for Hanning window function."""

    def test_hanning_basic(self):
        """Test basic Hanning window generation."""
        window = hanning_window(512)
        assert len(window) == 512
        assert window.dtype == np.float32

    def test_hanning_symmetry(self):
        """Test Hanning window is symmetric."""
        window = hanning_window(512)
        np.testing.assert_allclose(window[:256], window[-256:][::-1], rtol=1e-4, atol=1e-7)

    def test_hanning_peak(self):
        """Test Hanning window has peak at center."""
        window = hanning_window(512)
        assert window[256] == pytest.approx(1.0, abs=0.01)

    def test_hanning_edges_zero(self):
        """Test Hanning window has zeros at edges."""
        window = hanning_window(512)
        assert window[0] == pytest.approx(0.0, abs=1e-6)
        assert window[-1] == pytest.approx(0.0, abs=1e-6)

    def test_hanning_different_sizes(self):
        """Test Hanning window with different sizes."""
        for size in [64, 128, 256, 512, 1024]:
            window = hanning_window(size)
            assert len(window) == size
            assert window.max() == pytest.approx(1.0, abs=0.01)


class TestBlackmanWindow:
    """Tests for Blackman window function."""

    def test_blackman_basic(self):
        """Test basic Blackman window generation."""
        window = blackman_window(512)
        assert len(window) == 512
        assert window.dtype == np.float32

    def test_blackman_symmetry(self):
        """Test Blackman window is symmetric."""
        window = blackman_window(512)
        np.testing.assert_allclose(window[:256], window[-256:][::-1], rtol=1e-4, atol=1e-7)

    def test_blackman_peak(self):
        """Test Blackman window has peak at center."""
        window = blackman_window(512)
        assert window[256] == pytest.approx(1.0, abs=0.01)

    def test_blackman_edges_very_low(self):
        """Test Blackman window has very low values at edges."""
        window = blackman_window(512)
        # Blackman has better sidelobe suppression than Hamming/Hanning
        assert window[0] < 0.001
        assert window[-1] < 0.001

    def test_blackman_different_sizes(self):
        """Test Blackman window with different sizes."""
        for size in [64, 128, 256, 512, 1024]:
            window = blackman_window(size)
            assert len(window) == size
            assert window.max() == pytest.approx(1.0, abs=0.01)


class TestWindowComparison:
    """Tests comparing different window functions."""

    def test_windows_same_shape(self):
        """Test all windows have same shape."""
        size = 512
        hamming = hamming_window(size)
        hanning = hanning_window(size)
        blackman = blackman_window(size)

        assert len(hamming) == len(hanning) == len(blackman) == size

    def test_windows_edge_suppression(self):
        """Test windows have increasing edge suppression: Hamming < Hanning < Blackman."""
        size = 512
        hamming = hamming_window(size)
        hanning = hanning_window(size)
        blackman = blackman_window(size)

        # Check edge values (Blackman should be smallest)
        assert blackman[0] < hanning[0] < hamming[0]
        assert blackman[-1] < hanning[-1] < hamming[-1]

    def test_windows_normalization(self):
        """Test all windows are normalized to peak of 1."""
        size = 512
        for window_func in [hamming_window, hanning_window, blackman_window]:
            window = window_func(size)
            assert window.max() == pytest.approx(1.0, abs=0.01)
