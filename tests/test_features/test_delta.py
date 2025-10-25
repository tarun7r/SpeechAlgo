"""
Tests for speechalgo.features.delta module.
"""

import numpy as np
import pytest

from speechalgo.features.delta import DeltaFeatures


class TestDeltaFeatures:
    """Tests for DeltaFeatures class."""

    def test_initialization(self):
        """Test delta features initialization."""
        delta = DeltaFeatures(width=2)
        assert delta.width == 2
        assert delta.normalizer > 0

    def test_compute_delta(self):
        """Test delta computation on simple features."""
        # Create simple linearly increasing features
        features = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)

        delta = DeltaFeatures(width=1)
        delta_features = delta.compute_delta(features)

        # Check shape
        assert delta_features.shape == features.shape
        assert delta_features.dtype == np.float32

        # For linear increase, delta should be constant
        # (except at boundaries)
        assert np.all(delta_features[:, 1:-1] > 0)

    def test_compute_delta_constant(self):
        """Test delta of constant features is zero."""
        # Constant features
        features = np.ones((3, 10), dtype=np.float32)

        delta = DeltaFeatures(width=2)
        delta_features = delta.compute_delta(features)

        # Delta of constant should be near zero
        assert np.allclose(delta_features, 0.0, atol=1e-6)

    def test_compute_delta_delta(self):
        """Test delta-delta computation."""
        # Create features with quadratic change
        t = np.arange(20)
        features = np.array([t**2], dtype=np.float32)

        delta = DeltaFeatures(width=2)
        delta_delta = delta.compute_delta_delta(features)

        # Check shape
        assert delta_delta.shape == features.shape
        assert delta_delta.dtype == np.float32

    def test_delta_delta_via_double_delta(self):
        """Test delta-delta equals delta of delta."""
        features = np.random.randn(5, 20).astype(np.float32)

        delta = DeltaFeatures(width=2)

        # Method 1: compute_delta_delta
        delta_delta_1 = delta.compute_delta_delta(features)

        # Method 2: delta of delta
        delta_features = delta.compute_delta(features)
        delta_delta_2 = delta.compute_delta(delta_features)

        # Should be equal
        np.testing.assert_allclose(delta_delta_1, delta_delta_2)

    def test_different_widths(self):
        """Test delta computation with different widths."""
        features = np.random.randn(3, 20).astype(np.float32)

        for width in [1, 2, 3, 5]:
            delta = DeltaFeatures(width=width)
            delta_features = delta.compute_delta(features)

            assert delta_features.shape == features.shape

    def test_multivariate_features(self):
        """Test delta on multivariate features."""
        # Simulate MFCC features (13 coefficients x 100 frames)
        mfcc = np.random.randn(13, 100).astype(np.float32)

        delta = DeltaFeatures(width=2)
        mfcc_delta = delta.compute_delta(mfcc)
        mfcc_delta2 = delta.compute_delta_delta(mfcc)

        # All should have same shape
        assert mfcc_delta.shape == mfcc.shape
        assert mfcc_delta2.shape == mfcc.shape

        # Can stack them
        full_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        assert full_features.shape == (39, 100)

    def test_invalid_width(self):
        """Test initialization with invalid width."""
        with pytest.raises(ValueError):
            DeltaFeatures(width=0)

        with pytest.raises(ValueError):
            DeltaFeatures(width=-1)

    def test_invalid_features_shape(self):
        """Test with invalid features shape."""
        delta = DeltaFeatures(width=2)

        # 1D array should raise error
        features_1d = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        with pytest.raises(ValueError):
            delta.compute_delta(features_1d)

        # 3D array should raise error
        features_3d = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            delta.compute_delta(features_3d)

    def test_short_features(self):
        """Test delta on very short feature sequence."""
        # Only 3 frames
        features = np.array([[1, 2, 3]], dtype=np.float32)

        delta = DeltaFeatures(width=1)
        delta_features = delta.compute_delta(features)

        assert delta_features.shape == features.shape

    def test_repr(self):
        """Test string representation."""
        delta = DeltaFeatures(width=3)
        repr_str = repr(delta)
        assert "DeltaFeatures" in repr_str
        assert "3" in repr_str
