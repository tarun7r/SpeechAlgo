# SpeechAlgo

A comprehensive Python library providing clean, educational implementations of fundamental speech processing algorithms.

## Overview

SpeechAlgo provides reference implementations of 20 core speech processing algorithms, organized into five categories: preprocessing, voice activity detection (VAD), pitch detection, speech enhancement, and feature extraction. The library is designed for both educational purposes and production use, with clear code, comprehensive documentation, and mathematical foundations.

## Features

- **Clean, readable implementations** suitable for learning and research
- **Comprehensive documentation** with mathematical foundations and references
- **Type-annotated code** with extensive docstrings
- **NumPy-based** implementations for efficiency
- **Modular design** with consistent APIs
- **Well-tested** algorithms with unit tests
- **Real-time capable** for many algorithms

## Installation

### From PyPI (when published)

```bash
pip install speechalgo
```

### From source

```bash
git clone https://github.com/tarun7r/SpeechAlgo.git
cd SpeechAlgo
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Preprocessing

```python
import numpy as np
from speechalgo.preprocessing import hamming_window, FrameExtractor, MFCC
from speechalgo.utils import load_audio

# Load audio
audio, sr = load_audio('speech.wav', sample_rate=16000)

# Apply window function
window = hamming_window(512)
windowed = audio[:512] * window

# Extract overlapping frames
frame_extractor = FrameExtractor(frame_length=512, hop_length=256)
frames = frame_extractor.extract_frames(audio)
print(f"Extracted {len(frames)} frames")

# Extract MFCC features
mfcc = MFCC(sample_rate=16000, n_mfcc=13)
mfcc_features = mfcc.process(audio)
print(f"MFCC shape: {mfcc_features.shape}")  # (13, n_frames)
```

### Voice Activity Detection

```python
from speechalgo.vad import EnergyBasedVAD, SpectralEntropyVAD, ZeroCrossingVAD

# Energy-based VAD
energy_vad = EnergyBasedVAD(sample_rate=16000)
is_speech = energy_vad.process(audio)
print(f"Speech detected in {is_speech.sum()} / {len(is_speech)} frames")

# Spectral entropy VAD (more robust in noise)
entropy_vad = SpectralEntropyVAD(sample_rate=16000)
is_speech = entropy_vad.process(audio)

# Zero-crossing rate VAD
zcr_vad = ZeroCrossingVAD(sample_rate=16000)
is_speech = zcr_vad.process(audio)

# Extract speech segments
speech_frames = np.where(is_speech)[0]
if len(speech_frames) > 0:
    start = speech_frames[0] * 256  # hop_length
    end = speech_frames[-1] * 256
    speech_segment = audio[start:end]
```

### Pitch Detection

```python
from speechalgo.pitch import YIN, Autocorrelation, HPS

# YIN algorithm (recommended for speech)
yin = YIN(sample_rate=16000)
pitch = yin.estimate(audio_frame)
print(f"Estimated pitch: {pitch:.1f} Hz")

# Autocorrelation method
autocorr = Autocorrelation(sample_rate=16000)
pitch = autocorr.estimate(audio_frame)

# Harmonic Product Spectrum
hps = HPS(sample_rate=16000)
pitch = hps.estimate(audio_frame)
```

### Speech Enhancement

```python
from speechalgo.enhancement import SpectralSubtraction, WienerFilter, NoiseGate

# Spectral subtraction
enhancer = SpectralSubtraction(sample_rate=16000)
clean_audio = enhancer.process(noisy_audio, noise_profile)

# Wiener filtering
wiener = WienerFilter(sample_rate=16000)
clean_audio = wiener.process(noisy_audio)

# Noise gate
gate = NoiseGate(threshold=-40.0, sample_rate=16000)
gated_audio = gate.process(audio)
```

### Feature Extraction

```python
from speechalgo.features import SpectralFeatures, TemporalFeatures, DeltaFeatures

# Spectral features
spectral = SpectralFeatures(sample_rate=16000)
centroid = spectral.spectral_centroid(audio)
rolloff = spectral.spectral_rolloff(audio)
flux = spectral.spectral_flux(audio)

# Temporal features
temporal = TemporalFeatures(sample_rate=16000)
zcr = temporal.zero_crossing_rate(audio)
energy = temporal.short_time_energy(audio)

# Delta features (velocity and acceleration)
delta = DeltaFeatures()
mfcc_delta = delta.compute_delta(mfcc_features)
mfcc_delta2 = delta.compute_delta(mfcc_delta)
```

## Algorithm Categories

### Preprocessing (5 algorithms)

1. **Windowing** - Hamming, Hanning, Blackman window functions for spectral analysis
2. **Framing** - Overlapping frame extraction with configurable hop length
3. **Pre-emphasis** - High-frequency boosting filter (α=0.97)
4. **MFCC** - Mel-Frequency Cepstral Coefficients extraction
5. **Mel-Spectrogram** - Mel-scale frequency representation

### Voice Activity Detection (3 algorithms)

6. **Energy-based VAD** - Simple threshold-based detection using short-time energy
7. **Spectral Entropy VAD** - Entropy-based voice detection (robust in noise)
8. **Zero-Crossing VAD** - Combined energy and zero-crossing rate approach

### Pitch Detection (4 algorithms)

10. **Autocorrelation** - Classic time-domain pitch estimation
11. **YIN Algorithm** - Improved autocorrelation with difference function
12. **Cepstral Method** - Pitch detection using cepstrum
13. **Harmonic Product Spectrum (HPS)** - Frequency-domain approach

### Speech Enhancement (3 algorithms)

14. **Spectral Subtraction** - Classic noise reduction technique
15. **Wiener Filtering** - Statistical optimal filtering
16. **Noise Gate** - Threshold-based noise suppression

### Feature Extraction (4 algorithms)

17. **Spectral Features** - Centroid, rolloff, flux, bandwidth
18. **Zero Crossing Rate** - Time-domain feature extraction
19. **Short-Time Energy** - Energy computation in frames
20. **Delta Features** - First and second-order temporal derivatives

## Documentation

Comprehensive documentation with mathematical foundations and implementation notes:

- [Getting Started Tutorial](docs/tutorials/getting_started.md) - Installation and first steps
- [Preprocessing Theory](docs/theory/preprocessing.md) - Windowing, framing, MFCC, mel-spectrogram
- [VAD Theory](docs/theory/vad.md) - Voice activity detection methods and comparisons
- [Pitch Detection Theory](docs/theory/pitch.md) - Pitch estimation algorithms and best practices
- [Speech Enhancement Theory](docs/theory/enhancement.md) - Noise reduction techniques

## Examples

See [Getting Started Tutorial](docs/tutorials/getting_started.md) for comprehensive examples including:

- **Basic Examples**: Window functions, framing, pre-emphasis, MFCC, VAD, pitch detection, enhancement
- **Complete Workflows**:
  - Feature extraction pipeline (MFCC + delta + delta-delta = 39 dimensions)
  - VAD + pitch tracking for voiced speech analysis
  - Multi-stage noise reduction (noise gate → Wiener filter)
  - Multi-algorithm comparison and benchmarking

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SoundFile >= 0.11.0

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code passes all tests
- New features include tests
- Documentation is updated
- Code follows PEP 8 style guide
- Type hints are included

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This library implements algorithms from foundational research in speech processing:

- Davis & Mermelstein (1980) - MFCC
- de Cheveigné & Kawahara (2002) - YIN algorithm
- Rabiner & Schafer (1978) - Digital speech processing fundamentals
- Ephraim & Malah (1984) - Spectral subtraction
- And many others cited in individual algorithm documentation

## Project Status

**Current Version: 0.1.0 (MVP Complete)**

✅ **All 20 core algorithms implemented and tested!**


## Support

- **Issues**: [GitHub Issues](https://github.com/tarun7r/SpeechAlgo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tarun7r/SpeechAlgo/discussions)

