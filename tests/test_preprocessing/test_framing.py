"""
Tests for speechalgo.preprocessing.framing module.
"""

import numpy as np
import pytest

from speechalgo.preprocessing.framing import FrameExtractor


class TestFrameExtractor:
    """Tests for FrameExtractor class."""

    def test_frame_extractor_basic(self, sine_wave):
        """Test basic frame extraction."""
        extractor = FrameExtractor(frame_length=512, hop_length=256)
        frames = extractor.extract_frames(sine_wave)

        assert frames.ndim == 2
        assert frames.shape[1] == 512  # frame_length
        assert frames.dtype == np.float32

    def test_frame_count(self, sine_wave):
        """Test number of extracted frames."""
        frame_length = 512
        hop_length = 256
        extractor = FrameExtractor(frame_length=frame_length, hop_length=hop_length)
        frames = extractor.extract_frames(sine_wave)

        # With padding: audio is padded by frame_length//2 on each side
        # padded_length = len(audio) + 2 * (frame_length // 2) = len(audio) + frame_length
        padded_length = len(sine_wave) + frame_length
        expected_frames = 1 + (padded_length - frame_length) // hop_length
        assert frames.shape[0] == expected_frames

    def test_frame_overlap(self, sine_wave):
        """Test overlapping frames."""
        extractor = FrameExtractor(frame_length=512, hop_length=256, window=None)
        frames = extractor.extract_frames(sine_wave, pad=False)

        # Check that consecutive frames overlap
        if frames.shape[0] > 1:
            # Last 256 samples of frame 0 should match first 256 samples of frame 1
            overlap_samples = 512 - 256  # frame_length - hop_length
            np.testing.assert_array_equal(frames[0, overlap_samples:], frames[1, :overlap_samples])

    def test_no_overlap(self, sine_wave):
        """Test non-overlapping frames."""
        extractor = FrameExtractor(frame_length=512, hop_length=512)
        frames = extractor.extract_frames(sine_wave, pad=False)

        # With hop_length == frame_length, frames should not overlap
        expected_frames = len(sine_wave) // 512
        assert frames.shape[0] == expected_frames

    def test_with_window(self, sine_wave):
        """Test frame extraction with windowing."""
        extractor = FrameExtractor(frame_length=512, hop_length=256, window="hamming")
        frames = extractor.extract_frames(sine_wave, pad=False)

        assert frames.shape[1] == 512
        # Windowed frames should have attenuated edges compared to center
        # Check a middle frame
        if frames.shape[0] > 2:
            mid_frame = frames[frames.shape[0] // 2]
            assert np.abs(mid_frame[0]) < np.abs(mid_frame[256])

    def test_pad_end(self):
        """Test padding at the end of signal."""
        audio = np.ones(1000, dtype=np.float32)
        extractor = FrameExtractor(frame_length=512, hop_length=256)
        frames = extractor.extract_frames(audio, pad=True)

        # With padding, audio is extended
        assert frames.shape[0] >= 2

    def test_no_pad_end(self):
        """Test no padding at the end of signal."""
        audio = np.ones(1000, dtype=np.float32)
        extractor = FrameExtractor(frame_length=512, hop_length=256)
        frames = extractor.extract_frames(audio, pad=False)

        # Without padding, we should get floor((len - frame) / hop) + 1 frames
        expected_frames = (len(audio) - 512) // 256 + 1
        assert frames.shape[0] == expected_frames

    def test_short_audio(self):
        """Test frame extraction with audio shorter than frame length."""
        audio = np.ones(256, dtype=np.float32)
        extractor = FrameExtractor(frame_length=512, hop_length=256)
        
        # Audio shorter than frame_length should raise error
        with pytest.raises(ValueError, match="length"):
            frames = extractor.extract_frames(audio, pad=True)

    def test_tiny_audio(self, tiny_audio):
        """Test frame extraction with very short audio."""
        extractor = FrameExtractor(frame_length=512, hop_length=256)
        
        # Tiny audio (4 samples) is shorter than frame_length (512)
        # This should raise an error even with padding
        with pytest.raises(ValueError, match="length"):
            frames = extractor.extract_frames(tiny_audio, pad=True)

    def test_empty_audio(self):
        """Test frame extraction with empty audio."""
        audio = np.array([], dtype=np.float32)
        extractor = FrameExtractor(frame_length=512, hop_length=256)

        with pytest.raises(ValueError):
            extractor.extract_frames(audio)

    def test_different_hop_lengths(self, sine_wave):
        """Test frame extraction with different hop lengths."""
        for hop_length in [128, 256, 512]:
            extractor = FrameExtractor(frame_length=512, hop_length=hop_length)
            frames = extractor.extract_frames(sine_wave, pad=False)
            assert frames.shape[1] == 512

            # More overlap -> more frames
            expected_frames = (len(sine_wave) - 512) // hop_length + 1
            assert frames.shape[0] == expected_frames
