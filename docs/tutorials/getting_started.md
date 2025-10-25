# Getting Started with SpeechAlgo

Welcome to SpeechAlgo! This tutorial will help you get started with the library's speech processing algorithms.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Examples](#basic-examples)
4. [Module Overview](#module-overview)
5. [Common Workflows](#common-workflows)
6. [Next Steps](#next-steps)

---

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0

### Install from Source

```bash
git clone https://github.com/tarun7r/SpeechAlgo.git
cd SpeechAlgo
pip install -e .
```

### Development Installation

For development with testing tools:

```bash
pip install -e ".[dev]"
```

This installs additional dependencies:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- black (code formatter)
- flake8 (linter)

---

## Quick Start

### Test the Installation

```python
import speechalgo

print(f"SpeechAlgo version: {speechalgo.__version__}")
print("Available modules:", dir(speechalgo))
```

### Create Test Signal

```python
import numpy as np

# Create a 440 Hz sine wave (A4 note) at 16 kHz
sample_rate = 16000
duration = 1.0  # seconds
t = np.arange(int(sample_rate * duration)) / sample_rate
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

print(f"Audio shape: {audio.shape}")
print(f"Audio dtype: {audio.dtype}")
```

---

## Basic Examples

### 1. Windowing

Apply window functions to audio frames:

```python
from speechalgo.preprocessing import hamming_window, hanning_window

# Create windows
frame_length = 512
hamming = hamming_window(frame_length)
hanning = hanning_window(frame_length)

# Apply to audio frame
frame = audio[:frame_length]
windowed_hamming = frame * hamming
windowed_hanning = frame * hanning

print(f"Window shape: {hamming.shape}")
print(f"Hamming sum: {np.sum(hamming):.2f}")
```

### 2. Frame Extraction

Divide audio into overlapping frames:

```python
from speechalgo.preprocessing import FrameExtractor

# Initialize frame extractor
extractor = FrameExtractor(
    frame_length=512,  # 32 ms at 16 kHz
    hop_length=256,    # 50% overlap
    sample_rate=sample_rate
)

# Extract frames
frames = extractor.extract_frames(audio)

print(f"Frames shape: {frames.shape}")
print(f"Number of frames: {frames.shape[0]}")
print(f"Samples per frame: {frames.shape[1]}")
```

### 3. Pre-Emphasis

Boost high frequencies:

```python
from speechalgo.preprocessing import PreEmphasis

# Initialize pre-emphasis
pre_emphasis = PreEmphasis(coefficient=0.97)

# Apply pre-emphasis
emphasized = pre_emphasis.process(audio)

# Apply inverse (de-emphasis)
recovered = pre_emphasis.inverse(emphasized)

print(f"Emphasized shape: {emphasized.shape}")
print(f"Recovered matches original: {np.allclose(audio, recovered)}")
```

### 4. MFCC Features

Extract Mel-Frequency Cepstral Coefficients:

```python
from speechalgo.preprocessing import MFCC

# Initialize MFCC extractor
mfcc = MFCC(
    sample_rate=sample_rate,
    n_mfcc=13,          # Number of coefficients
    n_mels=40,          # Number of mel bands
    frame_length=512,
    hop_length=256
)

# Compute MFCCs
mfcc_features = mfcc.process(audio)

print(f"MFCC shape: {mfcc_features.shape}")
print(f"Shape is (n_frames, n_mfcc): {mfcc_features.shape}")
```

### 5. Voice Activity Detection

Detect speech vs. silence:

```python
from speechalgo.vad import EnergyBasedVAD

# Initialize VAD
vad = EnergyBasedVAD(
    sample_rate=sample_rate,
    frame_length=512,
    hop_length=256,
    threshold_db=-40
)

# Detect voice activity
is_speech = vad.process(audio)

print(f"VAD output shape: {is_speech.shape}")
print(f"Speech frames: {np.sum(is_speech)}")
print(f"Silence frames: {np.sum(~is_speech)}")
```

### 6. Pitch Detection

Estimate fundamental frequency (F0):

```python
from speechalgo.pitch import YIN

# Initialize pitch detector
yin = YIN(
    sample_rate=sample_rate,
    f0_min=80,      # Minimum F0 (Hz)
    f0_max=400      # Maximum F0 (Hz)
)

# Extract a single frame
frame = audio[:2048]  # ~128 ms

# Estimate pitch
f0 = yin.estimate(frame)

print(f"Estimated F0: {f0:.1f} Hz")
print(f"Expected F0: 440 Hz")
```

### 7. Noise Reduction

Enhance noisy speech:

```python
from speechalgo.enhancement import WienerFilter
import numpy as np

# Create noisy signal
noise = np.random.randn(len(audio)).astype(np.float32) * 0.1
noisy_audio = audio + noise

# Initialize Wiener filter
wiener = WienerFilter(
    sample_rate=sample_rate,
    frame_length=512,
    hop_length=256
)

# Estimate noise from initial segment
noise_segment = noisy_audio[:sample_rate//4]
wiener.estimate_noise(noise_segment)

# Process noisy audio
enhanced = wiener.process(noisy_audio)

print(f"Enhanced shape: {enhanced.shape}")
print(f"SNR improvement: {10*np.log10(np.var(audio)/np.var(enhanced-audio)):.1f} dB")
```

---

## Module Overview

### speechalgo.preprocessing

**Windowing:**
- `hamming_window()` - Hamming window function
- `hanning_window()` - Hanning window function
- `blackman_window()` - Blackman window function

**Framing:**
- `FrameExtractor` - Extract overlapping frames from audio

**Filtering:**
- `PreEmphasis` - High-pass pre-emphasis filter

**Features:**
- `MFCC` - Mel-Frequency Cepstral Coefficients
- `MelSpectrogram` - Mel-scale spectrogram

### speechalgo.vad

**Algorithms:**
- `EnergyBasedVAD` - Energy threshold-based detection
- `SpectralEntropyVAD` - Spectral entropy-based detection
- `ZeroCrossingVAD` - Zero-crossing rate-based detection

### speechalgo.pitch

**Algorithms:**
- `Autocorrelation` - Autocorrelation-based pitch detection
- `YIN` - YIN algorithm (robust, recommended)
- `CepstralPitch` - Cepstrum-based pitch detection
- `HPS` - Harmonic Product Spectrum

### speechalgo.enhancement

**Algorithms:**
- `SpectralSubtraction` - Spectral subtraction noise reduction
- `WienerFilter` - Wiener filtering (optimal MMSE)
- `NoiseGate` - Simple threshold-based noise gate

### speechalgo.features

**Temporal Features:**
- `zero_crossing_rate()` - Count zero crossings per frame
- `energy()` - Frame energy computation
- `rms_energy()` - RMS energy per frame

**Spectral Features:**
- `spectral_centroid()` - Center of mass of spectrum
- `spectral_rolloff()` - Frequency below which X% of energy lies
- `spectral_flux()` - Change in spectrum over time
- `spectral_bandwidth()` - Spread of spectrum around centroid

**Delta Features:**
- `delta_features()` - First-order differences (velocity)
- `delta_delta_features()` - Second-order differences (acceleration)

---

## Common Workflows

### Workflow 1: Feature Extraction for Recognition

```python
from speechalgo.preprocessing import MFCC, PreEmphasis
from speechalgo.features import delta_features, delta_delta_features

# Load audio (assume you have it)
audio = ...  # shape: (n_samples,)

# 1. Pre-emphasis
pre_emphasis = PreEmphasis(coefficient=0.97)
emphasized = pre_emphasis.process(audio)

# 2. Extract MFCCs
mfcc = MFCC(sample_rate=16000, n_mfcc=13)
mfcc_features = mfcc.process(emphasized)

# 3. Compute delta features
delta = delta_features(mfcc_features)
delta_delta = delta_delta_features(mfcc_features)

# 4. Concatenate all features
features = np.hstack([mfcc_features, delta, delta_delta])

print(f"Final feature shape: {features.shape}")
# Output: (n_frames, 39) -> 13 MFCC + 13 Δ + 13 ΔΔ
```

### Workflow 2: Voice Activity Detection + Pitch Tracking

```python
from speechalgo.vad import EnergyBasedVAD
from speechalgo.pitch import YIN
from speechalgo.preprocessing import FrameExtractor

# 1. Extract frames
extractor = FrameExtractor(frame_length=2048, hop_length=512)
frames = extractor.extract_frames(audio)

# 2. Detect voice activity
vad = EnergyBasedVAD(sample_rate=16000)
is_speech = vad.process(audio)

# 3. Estimate pitch only on voiced frames
yin = YIN(sample_rate=16000)
pitch = np.zeros(len(frames))

for i, frame in enumerate(frames):
    if is_speech[i]:  # Only process speech frames
        pitch[i] = yin.estimate(frame)

print(f"Voiced frames: {np.sum(is_speech)}/{len(is_speech)}")
print(f"Mean F0 (voiced): {np.mean(pitch[pitch > 0]):.1f} Hz")
```

### Workflow 3: Noise Reduction Pipeline

```python
from speechalgo.vad import EnergyBasedVAD
from speechalgo.enhancement import WienerFilter, NoiseGate

# 1. Apply noise gate (remove silence)
noise_gate = NoiseGate(sample_rate=16000, threshold_db=-50)
gated = noise_gate.process(noisy_audio)

# 2. Estimate noise from silence periods
vad = EnergyBasedVAD(sample_rate=16000)
is_speech = vad.process(gated)

# Extract noise-only frames
noise_indices = np.where(~is_speech)[0]
if len(noise_indices) > 0:
    noise_samples = []
    extractor = FrameExtractor(frame_length=512, hop_length=256)
    frames = extractor.extract_frames(gated)
    for idx in noise_indices[:10]:  # Use first 10 noise frames
        noise_samples.append(frames[idx])
    noise_estimate = np.concatenate(noise_samples)
else:
    # Fallback: use first 0.5 seconds
    noise_estimate = gated[:sample_rate//2]

# 3. Apply Wiener filter
wiener = WienerFilter(sample_rate=16000)
wiener.estimate_noise(noise_estimate)
enhanced = wiener.process(gated)

print(f"Noise reduction complete")
print(f"Output shape: {enhanced.shape}")
```

### Workflow 4: Multi-Algorithm Comparison

```python
from speechalgo.pitch import Autocorrelation, YIN, CepstralPitch, HPS

# Initialize all pitch detectors
algorithms = {
    'Autocorrelation': Autocorrelation(sample_rate=16000),
    'YIN': YIN(sample_rate=16000),
    'Cepstral': CepstralPitch(sample_rate=16000),
    'HPS': HPS(sample_rate=16000),
}

# Extract a voiced frame
frame = audio[5000:7048]  # 2048 samples

# Compare algorithms
results = {}
for name, algorithm in algorithms.items():
    f0 = algorithm.estimate(frame)
    results[name] = f0
    print(f"{name:15s}: {f0:6.1f} Hz")

# Compute statistics
estimates = [f0 for f0 in results.values() if f0 > 0]
print(f"\nMean: {np.mean(estimates):.1f} Hz")
print(f"Std:  {np.std(estimates):.1f} Hz")
```

---

## Next Steps

### Learn More

1. **Theory Documentation:**
   - `docs/theory/preprocessing.md` - Windowing, MFCC, mel-scale
   - `docs/theory/vad.md` - VAD algorithms and comparison
   - `docs/theory/pitch.md` - Pitch detection methods
   - `docs/theory/enhancement.md` - Noise reduction principles

2. **API Documentation:**
   - Check docstrings in source code
   - All functions have detailed documentation
   - Examples included in docstrings

3. **Test Suite:**
   - `tests/` directory contains comprehensive examples
   - Run tests: `pytest tests/ -v`
   - See test coverage: `pytest tests/ --cov=speechalgo`

### Common Issues

**Import errors:**
```python
# Make sure you're importing from the right module
from speechalgo.preprocessing import MFCC  # Correct
from speechalgo.MFCC import MFCC           # Wrong
```

**Audio format:**
```python
# Ensure audio is float32 numpy array
audio = audio.astype(np.float32)

# Normalize if needed (0-1 range)
audio = audio / np.max(np.abs(audio))
```

**Frame length for pitch:**
```python
# Pitch detectors need longer frames
frame_length = 2048  # ~128 ms at 16 kHz
# Contains multiple pitch periods for reliable estimation
```

### Best Practices

1. **Always use float32** for audio data
2. **Normalize audio** to prevent overflow
3. **Check sample rate** matches algorithm expectations
4. **Use appropriate frame sizes** for different algorithms
5. **Combine algorithms** for robust results (e.g., VAD + pitch)

### Getting Help

- **Documentation:** Check theory docs and docstrings
- **Examples:** Review test files for usage patterns
- **Issues:** Report bugs on GitHub
- **Source Code:** All algorithms are well-documented

---

## Summary

You've learned:

✅ How to install SpeechAlgo  
✅ Basic usage of each module  
✅ Common processing workflows  
✅ Best practices and tips

**Next:** Explore the theory documentation to understand the algorithms in depth, or dive into the source code to see implementation details.

Happy signal processing! 🎵
