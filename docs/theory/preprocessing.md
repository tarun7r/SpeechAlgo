# Audio Preprocessing Theory

This document provides the mathematical foundations and theoretical concepts behind the preprocessing algorithms implemented in SpeechAlgo.

## Table of Contents

1. [Windowing](#windowing)
2. [Framing](#framing)
3. [Pre-Emphasis](#pre-emphasis)
4. [Mel-Spectrogram](#mel-spectrogram)
5. [MFCC (Mel-Frequency Cepstral Coefficients)](#mfcc)

---

## Windowing

### Purpose

Windowing is applied to audio frames to reduce spectral leakage when computing the Fourier transform. Without windowing, abrupt discontinuities at frame boundaries create artificial high-frequency components in the spectrum.

### Mathematical Foundation

A window function w[n] is multiplied element-wise with the signal frame:

```
x_windowed[n] = x[n] * w[n],  n = 0, 1, ..., N-1
```

where N is the frame length.

### Common Window Functions

#### 1. Rectangular (Boxcar) Window

```
w[n] = 1,  for all n
```

**Properties:**
- Narrowest main lobe (best frequency resolution)
- Highest side lobes (worst leakage suppression)
- Only suitable when signal is exactly periodic within the frame

#### 2. Hamming Window

```
w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
```

**Properties:**
- Main lobe width: 8π/N
- First side lobe: -43 dB
- Good balance between resolution and leakage
- Most commonly used in speech processing

**Trade-offs:**
- Wider main lobe than rectangular (reduced frequency resolution)
- Much lower side lobes (better leakage suppression)

#### 3. Hanning Window

```
w[n] = 0.5 - 0.5 * cos(2πn / (N-1))
```

**Properties:**
- Main lobe width: 8π/N (same as Hamming)
- First side lobe: -32 dB (better than Hamming)
- Falls to zero at endpoints (smoother than Hamming)

**Difference from Hamming:**
- Slightly better side lobe suppression
- Slightly worse main lobe width
- Better for closely spaced frequency components

#### 4. Blackman Window

```
w[n] = 0.42 - 0.5 * cos(2πn / (N-1)) + 0.08 * cos(4πn / (N-1))
```

**Properties:**
- Main lobe width: 12π/N (widest)
- First side lobe: -58 dB (best suppression)
- Excellent for high dynamic range applications

**Trade-offs:**
- Much wider main lobe (poorest frequency resolution)
- Excellent side lobe suppression
- Best when accuracy of amplitude is critical

### When to Use Which Window

| Window | Use Case |
|--------|----------|
| **Rectangular** | Signal is exactly periodic in frame, need maximum resolution |
| **Hamming** | General speech processing, balanced performance |
| **Hanning** | Closely spaced frequency components, need smooth tapering |
| **Blackman** | High dynamic range, accurate amplitude measurement |

### Spectral Leakage

Without windowing, the DFT assumes the signal is periodic with period N. If the signal doesn't satisfy this, discontinuities at frame boundaries create spurious frequency components:

```
Energy leaks from main frequency into neighboring bins
```

**Effect of Windowing:**
- Reduces discontinuities at frame boundaries
- Trades frequency resolution for reduced leakage
- Smooths spectrum at the cost of wider peaks

---

## Framing

### Purpose

Audio signals are non-stationary (properties change over time). Framing divides the signal into short segments where properties can be assumed stationary (typically 20-40 ms for speech).

### Frame Extraction

```
frame[i, :] = audio[i * hop_length : i * hop_length + frame_length]
```

where:
- `frame_length`: Number of samples per frame (typical: 512, 1024)
- `hop_length`: Number of samples between consecutive frames (typical: frame_length / 2)

### Overlap Percentage

```
overlap = 1 - (hop_length / frame_length)
```

**Common values:**
- 50% overlap (hop_length = frame_length / 2): Most common
- 75% overlap (hop_length = frame_length / 4): More temporal resolution
- 0% overlap (hop_length = frame_length): Fast, low redundancy

### Frame Parameters Selection

#### Frame Length

**Short frames (256 samples, ~16ms at 16kHz):**
- Better temporal resolution
- Worse frequency resolution
- May violate stationarity assumption for some features
- Good for: Fast-changing signals, real-time applications

**Long frames (1024 samples, ~64ms at 16kHz):**
- Better frequency resolution
- Worse temporal resolution
- Better for pitch detection (need multiple periods)
- Good for: Harmonic analysis, detailed spectral features

**Rule of thumb for speech:**
- 20-30 ms frames (320-480 samples at 16kHz)
- Captures 2-3 pitch periods for typical voices
- Short enough to assume stationarity

#### Hop Length

**Small hop (high overlap):**
- More frames → more computation
- Smoother temporal evolution
- Better for tracking rapid changes
- Good for: Pitch tracking, VAD

**Large hop (low overlap):**
- Fewer frames → faster processing
- May miss rapid transients
- Good for: Batch processing, less critical timing

### Padding

When signal length is not a multiple of hop_length:

**Pad end:**
```
Pad signal with zeros to complete last frame
```

**Drop last:**
```
Discard partial frame at the end
```

**Recommendation:** Pad end for most applications to preserve all information.

---

## Pre-Emphasis

### Purpose

Pre-emphasis amplifies high frequencies to:
1. Balance the spectrum (speech naturally has more energy at low frequencies)
2. Improve SNR for high frequencies
3. Make formants more visible in spectrum
4. Compensate for high-frequency roll-off in recording equipment

### Mathematical Formulation

Pre-emphasis is a first-order high-pass FIR filter:

```
y[n] = x[n] - α * x[n-1]
```

where α is the pre-emphasis coefficient (typical: 0.95-0.97).

### Frequency Response

```
H(z) = 1 - α * z^{-1}

|H(f)| = sqrt(1 + α^2 - 2α * cos(2πf/fs))
```

where fs is the sample rate.

**Effect:**
- At DC (f=0): |H(0)| = 1 - α ≈ 0.05 (strong attenuation)
- At Nyquist (f=fs/2): |H(fs/2)| = 1 + α ≈ 1.95 (amplification)

### Why High Frequencies Need Boosting

1. **Natural speech spectrum:** Energy decreases ~6 dB/octave
2. **Perceptual importance:** High frequencies carry consonant information
3. **Recording artifacts:** Microphones often attenuate high frequencies
4. **Processing robustness:** Balanced spectrum improves many algorithms

### De-Emphasis (Inverse)

To recover the original signal:

```
x[n] = y[n] + α * x[n-1]
```

This is crucial after processing if natural-sounding audio is desired.

### Coefficient Selection

| α | Effect |
|---|--------|
| **0.0** | No pre-emphasis (bypass) |
| **0.95** | Mild emphasis (conservative) |
| **0.97** | Standard emphasis (most common) |
| **0.99** | Strong emphasis (aggressive) |

**Recommendation:** Start with α = 0.97 for speech processing.

### Limitations

- Can amplify high-frequency noise
- May cause numerical issues if signal has high-frequency content
- Not needed if working directly with power spectral features

---

## Mel-Spectrogram

### Purpose

Convert linear-frequency spectrogram to mel-frequency scale, which better represents human auditory perception.

### Mel Scale

The mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.

**Mel-to-Hz conversion:**

```
f_mel = 2595 * log10(1 + f_hz / 700)
```

**Hz-to-mel conversion:**

```
f_hz = 700 * (10^(f_mel / 2595) - 1)
```

### Why Mel Scale?

Human hearing is approximately linear below 1 kHz and logarithmic above:
- Equal differences in Hz are not perceived equally
- Low frequencies: Can distinguish 50 Hz from 100 Hz
- High frequencies: Cannot distinguish 5000 Hz from 5050 Hz

### Mel Filterbank

A set of triangular filters spaced according to the mel scale:

```
Filter m: Triangular shape from f[m-1] to f[m] to f[m+1]
```

**Properties:**
- Filters overlap by 50%
- Narrower at low frequencies (better resolution)
- Wider at high frequencies (perceptual relevance)
- Typically 40-80 filters for speech

### Computation Steps

1. **Compute power spectrogram:**
   ```
   S[t, f] = |STFT(x)[t, f]|^2
   ```

2. **Apply mel filterbank:**
   ```
   M[t, m] = sum_f ( S[t, f] * FilterBank[m, f] )
   ```

3. **Convert to dB (optional):**
   ```
   M_dB[t, m] = 10 * log10(M[t, m] + ε)
   ```

### Parameters

**Number of mel bands:**
- 40 bands: Minimal representation
- 80 bands: Standard for speech
- 128 bands: High resolution for music

**Frequency range:**
- Speech: 0-8000 Hz (most energy below 8 kHz)
- Music: 0-22050 Hz (full range for 44.1 kHz audio)

---

## MFCC (Mel-Frequency Cepstral Coefficients)

### Purpose

MFCCs are the most widely used features in speech processing:
- Compact representation (13-26 coefficients)
- Capture spectral envelope (formant structure)
- Robust to noise and speaker variation
- Standard in speech recognition

### Mathematical Foundation

MFCCs separate the vocal tract filter from the excitation source using the cepstrum concept.

**Cepstrum definition:**
```
c[n] = IDFT(log(|DFT(x[n])|))
```

The term "cepstrum" is an anagram of "spectrum" (quefrency ↔ frequency).

### Computation Pipeline

```
Audio → Framing → Windowing → FFT → Mel Filterbank → Log → DCT → MFCCs
```

#### Step-by-step:

1. **Frame the signal:**
   ```
   frames = extract_frames(audio, frame_length, hop_length)
   ```

2. **Apply window:**
   ```
   windowed = frames * window
   ```

3. **Compute power spectrum:**
   ```
   spectrum = |FFT(windowed)|^2
   ```

4. **Apply mel filterbank:**
   ```
   mel_spectrum = mel_filterbank @ spectrum^T
   ```

5. **Take logarithm:**
   ```
   log_mel = log(mel_spectrum + ε)
   ```

6. **Apply DCT:**
   ```
   mfcc = DCT(log_mel)
   ```

### Why DCT?

The Discrete Cosine Transform decorrelates the mel filterbank outputs:
- Mel filterbank outputs are correlated (overlapping filters)
- DCT provides orthogonal basis
- First few coefficients capture most information
- Compression: Keep first 13-26 coefficients

### MFCC Coefficients

**Coefficient 0 (C0):**
- Represents overall signal energy
- Often excluded in recognition (normalization)
- Useful for energy-based features

**Coefficients 1-12:**
- Capture spectral shape (formant structure)
- Most discriminative for speech
- Core features for recognition

**Coefficients 13-25:**
- Capture fine spectral details
- Less critical but can improve accuracy
- Higher dimensional → more data needed

### Liftering

Liftering applies a window to cepstral coefficients to emphasize certain quefrencies:

```
mfcc_liftered[n] = mfcc[n] * (1 + (L/2) * sin(πn / L))
```

where L is the liftering parameter (typical: 22).

**Purpose:**
- De-emphasizes high quefrencies (rapid spectral changes)
- Emphasizes low quefrencies (slow spectral changes)
- Improves robustness to noise

### Delta Features

First-order differences (delta or velocity):

```
Δmfcc[t] = (mfcc[t+1] - mfcc[t-1]) / 2
```

Second-order differences (delta-delta or acceleration):

```
ΔΔmfcc[t] = (Δmfcc[t+1] - Δmfcc[t-1]) / 2
```

**Purpose:**
- Capture temporal dynamics
- Improve speech recognition accuracy
- Typical features: 13 MFCC + 13 Δ + 13 ΔΔ = 39 dimensions

### Parameter Selection

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| Sample rate | 16 kHz | Standard for speech |
| Frame length | 512 (32 ms) | Captures 2-3 pitch periods |
| Hop length | 256 (16 ms) | 50% overlap |
| n_mfcc | 13 | Compact, standard |
| n_mfcc | 26 | More detailed |
| n_mels | 40-80 | Mel filterbank resolution |
| Lifter | 22 | Standard value |

### Applications

1. **Speech Recognition:** Primary feature set
2. **Speaker Recognition:** With additional features (pitch, energy)
3. **Emotion Recognition:** Combined with prosodic features
4. **Audio Classification:** Music genre, sound events

### Advantages

- Compact representation (13 coefficients vs. 256+ spectral bins)
- Perceptually motivated (mel scale)
- Robust to additive noise
- Separates excitation from vocal tract
- Well-studied with decades of research

### Limitations

- Information loss (dimensionality reduction)
- Sensitive to channel effects (telephone, microphone)
- Assumes short-term stationarity
- May not capture fine spectral details

---

## References

1. Rabiner, L. R., & Schafer, R. W. (2007). Introduction to digital speech processing.
2. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. IEEE TASSP, 28(4), 357-366.
3. O'Shaughnessy, D. (2000). Speech communications: human and machine. IEEE press.
4. Gold, B., Morgan, N., & Ellis, D. (2011). Speech and audio signal processing. John Wiley & Sons.
5. Oppenheim, A. V., & Schafer, R. W. (1989). Discrete-time signal processing. Prentice Hall.
