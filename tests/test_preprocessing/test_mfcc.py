"""
Tests for speechalgo.preprocessing.mfcc module.
"""

import numpy as np

from speechalgo.preprocessing.mfcc import MFCC


class TestMFCC:
    """Tests for MFCC class."""

    def test_initialization(self, sample_rate):
        """Test MFCC initialization."""
        mfcc = MFCC(sample_rate=sample_rate)
        assert mfcc.sample_rate == sample_rate
        assert mfcc.n_mfcc > 0
        assert mfcc.n_mels > 0
        assert mfcc.n_fft > 0

    def test_process_sine_wave(self, sine_wave, sample_rate):
        """Test MFCC on sine wave."""
        mfcc = MFCC(sample_rate=sample_rate, n_mfcc=13)
        features = mfcc.process(sine_wave)

        # Check shape: (n_mfcc, n_frames)
        assert features.ndim == 2
        assert features.shape[0] == 13
        assert features.shape[1] > 0
        assert features.dtype == np.float32

    def test_different_n_mfcc(self, sine_wave, sample_rate):
        """Test MFCC with different number of coefficients."""
        for n_mfcc in [12, 13, 20, 40]:
            mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
            features = mfcc.process(sine_wave)

            assert features.shape[0] == n_mfcc

    def test_different_n_mels(self, sine_wave, sample_rate):
        """Test MFCC with different number of mel bands."""
        for n_mels in [20, 40, 80]:
            mfcc = MFCC(sample_rate=sample_rate, n_mfcc=13, n_mels=n_mels)
            features = mfcc.process(sine_wave)

            assert features.shape[0] == 13

    def test_with_liftering(self, sine_wave, sample_rate):
        """Test MFCC with liftering."""
        mfcc_no_lift = MFCC(sample_rate=sample_rate, lifter=0)
        mfcc_with_lift = MFCC(sample_rate=sample_rate, lifter=22)

        features_no_lift = mfcc_no_lift.process(sine_wave)
        features_with_lift = mfcc_with_lift.process(sine_wave)

        # Liftering should modify the features
        assert not np.allclose(features_no_lift, features_with_lift)

    def test_first_coefficient_energy(self, sine_wave, sample_rate):
        """Test that first MFCC coefficient represents energy."""
        mfcc = MFCC(sample_rate=sample_rate, n_mfcc=13)
        features = mfcc.process(sine_wave)

        # First coefficient (C0) should be related to log energy
        # and should be relatively large compared to others
        c0 = features[0, :]
        assert np.all(np.isfinite(c0))

    def test_silence(self, zero_audio, sample_rate):
        """Test MFCC on silence."""
        mfcc = MFCC(sample_rate=sample_rate)
        features = mfcc.process(zero_audio)

        # Should produce features (though may be small/negative for silence)
        assert features.shape[0] == 13
        assert features.shape[1] > 0

    def test_speech_like_signal(self, speech_like_signal, sample_rate):
        """Test MFCC on speech-like signal."""
        mfcc = MFCC(sample_rate=sample_rate)
        features = mfcc.process(speech_like_signal)

        # Should produce reasonable features
        assert features.shape[0] == 13
        assert features.shape[1] > 0

        # Features should vary over time for speech
        variance = np.var(features, axis=1)
        assert np.any(variance > 0)

    def test_frame_parameters(self, sine_wave, sample_rate):
        """Test MFCC with different frame parameters."""
        mfcc = MFCC(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
        )
        features = mfcc.process(sine_wave)

        assert features.shape[0] == 13
        assert features.shape[1] > 0

    def test_frequency_range(self, sine_wave, sample_rate):
        """Test MFCC with custom frequency range."""
        mfcc = MFCC(
            sample_rate=sample_rate,
            freq_min=80.0,
            freq_max=4000.0,
        )
        features = mfcc.process(sine_wave)

        assert features.shape[0] == 13

    def test_tiny_audio(self, tiny_audio, sample_rate):
        """Test MFCC with very short audio."""
        # Tiny audio (4 samples) is too short, expect it to raise ValueError
        mfcc = MFCC(sample_rate=sample_rate)
        with np.testing.assert_raises(ValueError):
            features = mfcc.process(tiny_audio)

    def test_deterministic(self, sine_wave, sample_rate):
        """Test MFCC is deterministic."""
        mfcc = MFCC(sample_rate=sample_rate)
        features1 = mfcc.process(sine_wave)
        features2 = mfcc.process(sine_wave)

        np.testing.assert_array_equal(features1, features2)

    def test_output_dtype(self, sine_wave, sample_rate):
        """Test MFCC output is float32."""
        mfcc = MFCC(sample_rate=sample_rate)
        features = mfcc.process(sine_wave)

        assert features.dtype == np.float32

    def test_repr(self, sample_rate):
        """Test string representation."""
        mfcc = MFCC(sample_rate=sample_rate)
        repr_str = repr(mfcc)
        assert "MFCC" in repr_str
        assert str(sample_rate) in repr_str
