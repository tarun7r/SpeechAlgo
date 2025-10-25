# Voice Activity Detection (VAD) Theory

This document covers the theoretical foundations of Voice Activity Detection algorithms implemented in SpeechAlgo.

## Table of Contents

1. [VAD Problem Formulation](#vad-problem-formulation)
2. [Energy-Based VAD](#energy-based-vad)
3. [Spectral Entropy VAD](#spectral-entropy-vad)
4. [Zero-Crossing Rate VAD](#zero-crossing-rate-vad)
5. [Performance Comparison](#performance-comparison)
6. [Best Practices](#best-practices)

---

## VAD Problem Formulation

### Definition

Voice Activity Detection (VAD) is the task of classifying audio frames as either:
- **Speech (voiced):** Contains speech signal
- **Non-speech (unvoiced):** Silence, noise, or non-speech audio

### Mathematical Formulation

Given audio signal x[n], divide into frames and classify each frame:

```
VAD: x[t] → {0, 1}

where:
  0 = non-speech (silence/noise)
  1 = speech
  t = frame index
```

### Applications

1. **Speech Recognition:** Skip non-speech frames (save computation)
2. **Speech Coding:** Allocate bits only to speech frames
3. **Telephony:** Comfort noise generation during silence
4. **Echo Cancellation:** Update filters only during speech
5. **Speaker Diarization:** Segment audio by speaker activity

### Challenges

1. **Low SNR:** Speech barely above noise level
2. **Non-stationary noise:** Traffic, babble, music
3. **Reverberation:** Echoes blur speech boundaries
4. **Unvoiced consonants:** Low energy but still speech
5. **Real-time constraints:** Fast processing required

### Performance Metrics

**Confusion Matrix:**

```
                Predicted
                Speech  Silence
Actual  Speech    TP      FN
        Silence   FP      TN
```

**Key Metrics:**

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Trade-offs:**
- High threshold → More false negatives (miss speech)
- Low threshold → More false positives (flag noise as speech)

---

## Energy-Based VAD

### Principle

Speech frames have significantly higher energy than silence/noise frames. Compare frame energy to a threshold:

```
E[t] = sum( x[t, n]^2 ) for n in frame

VAD[t] = 1 if E[t] > threshold
         0 otherwise
```

### Short-Term Energy

**Computation:**

```
E[t] = (1/N) * sum_{n=0}^{N-1} (x[t*hop + n])^2
```

where N is frame length.

**In dB:**

```
E_dB[t] = 10 * log10(E[t] + ε)
```

where ε is a small constant to avoid log(0).

### Adaptive Thresholding

Fixed thresholds fail in varying noise conditions. Use adaptive threshold:

```
threshold = μ_noise + k * σ_noise
```

where:
- μ_noise: Mean energy of noise-only frames
- σ_noise: Standard deviation of noise energy
- k: Sensitivity factor (typical: 2-3)

**Noise Estimation:**
- Assume first few frames are noise-only
- Update during detected silence periods
- Use minimum statistics

### Advantages

✅ Simple and fast  
✅ Low computational cost  
✅ Works well in clean conditions  
✅ No frequency analysis needed

### Limitations

❌ Fails in low SNR (speech energy ≈ noise energy)  
❌ Sensitive to background noise level  
❌ May miss unvoiced consonants (low energy)  
❌ Requires threshold tuning per environment

### When to Use

- Clean or high SNR environments (> 20 dB)
- Controlled recording conditions
- Real-time applications (minimal latency)
- Preliminary VAD before more sophisticated methods

---

## Spectral Entropy VAD

### Principle

Speech has structured spectrum (harmonics, formants) with low entropy. Noise has flat, random spectrum with high entropy.

```
Speech: Energy concentrated in few frequencies → Low entropy
Noise: Energy spread across all frequencies → High entropy
```

### Shannon Entropy

For a discrete probability distribution P = {p₁, p₂, ..., pₙ}:

```
H(P) = -sum( pᵢ * log(pᵢ) )
```

### Spectral Entropy Computation

1. **Compute power spectrum:**
   ```
   S[f] = |FFT(frame)|^2
   ```

2. **Normalize to probability distribution:**
   ```
   P[f] = S[f] / sum(S)
   ```

3. **Compute entropy:**
   ```
   H = -sum( P[f] * log(P[f]) )
   ```

4. **Normalize entropy:**
   ```
   H_norm = H / log(N_freq)
   ```
   
   where N_freq is number of frequency bins.

**Range:** H_norm ∈ [0, 1]
- 0: All energy in one frequency (pure tone)
- 1: Energy equally distributed (white noise)

### Threshold-Based Decision

```
VAD[t] = 1 if H_norm[t] < threshold
         0 otherwise
```

**Typical threshold:** 0.7-0.8 for clean speech

### Advantages

✅ Robust to amplitude variations  
✅ Works better in noise than energy-based  
✅ Captures spectral structure  
✅ Less sensitive to absolute energy level

### Limitations

❌ Requires FFT computation  
❌ May confuse music/tonal noise with speech  
❌ Threshold still needs tuning  
❌ Slower than energy-based

### When to Use

- Moderate SNR environments (10-20 dB)
- When noise level varies
- When spectral characteristics differ between speech and noise
- When energy-based VAD fails

---

## Zero-Crossing Rate VAD

### Principle

Zero-crossing rate (ZCR) measures how often the signal crosses zero amplitude:

```
ZCR = (1/N) * sum_{n=0}^{N-2} |sgn(x[n+1]) - sgn(x[n])|
```

where sgn(x) = 1 if x ≥ 0, -1 otherwise.

### Relationship to Frequency Content

**High ZCR:**
- High-frequency content (many zero crossings)
- Unvoiced consonants: /s/, /f/, /th/
- Noise

**Low ZCR:**
- Low-frequency content (few zero crossings)
- Voiced speech: vowels
- Silence (near zero)

### Spectral Interpretation

ZCR approximates the dominant frequency:

```
f_approx = ZCR * sample_rate / 2
```

Not accurate but gives rough frequency information without FFT.

### VAD Using ZCR

**Single threshold:**
```
VAD[t] = 1 if ZCR[t] > threshold
         0 otherwise
```

**Dual threshold (better):**
```
VAD[t] = 1 if ZCR_min < ZCR[t] < ZCR_max
         0 otherwise
```

This captures both voiced (low ZCR) and unvoiced (high ZCR) speech.

### Combined with Energy

ZCR alone is insufficient. Combine with energy:

```
VAD[t] = 1 if (Energy[t] > E_thresh) AND (ZCR_min < ZCR[t] < ZCR_max)
         0 otherwise
```

### Advantages

✅ Very fast (no FFT needed)  
✅ Detects unvoiced consonants better than energy  
✅ Complements energy-based features  
✅ Useful for distinguishing voiced/unvoiced speech

### Limitations

❌ Not robust to noise alone  
❌ Needs combination with other features  
❌ Thresholds depend on signal characteristics  
❌ Background noise can have high ZCR

### When to Use

- Combined feature with energy
- Detecting unvoiced speech segments
- Classifying voiced vs. unvoiced within speech
- Fast preliminary detection

---

## Performance Comparison

### SNR Dependency

| Method | Clean (>20 dB) | Moderate (10-20 dB) | Low (<10 dB) |
|--------|----------------|---------------------|--------------|
| **Energy-Based** | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐ Poor |
| **Spectral Entropy** | ⭐⭐⭐ Excellent | ⭐⭐⭐ Excellent | ⭐⭐ Good |
| **Zero-Crossing** | ⭐⭐ Good | ⭐⭐ Good | ⭐ Poor |

### Computational Cost

| Method | Operations | Latency | Memory |
|--------|-----------|---------|---------|
| **Energy-Based** | O(N) | Minimal | Minimal |
| **Spectral Entropy** | O(N log N) | Low | Moderate |
| **Zero-Crossing** | O(N) | Minimal | Minimal |

### Noise Type Robustness

| Method | White Noise | Babble | Music | Traffic |
|--------|-------------|---------|-------|---------|
| **Energy-Based** | Poor | Poor | Poor | Poor |
| **Spectral Entropy** | Good | Good | Poor | Good |
| **Zero-Crossing** | Poor | Poor | Poor | Moderate |

### Feature Characteristics

```
Energy:           ████████              Speech high, noise low
Spectral Entropy: ▓▓▓▓░░░░              Speech low, noise high  
Zero-Crossing:    ▓░░░▓░░░              Voiced low, unvoiced high
```

---

## Best Practices

### 1. Method Selection

**Clean environments:**
```python
# Energy-based is sufficient
vad = EnergyBasedVAD(sample_rate=16000, threshold_db=-40)
```

**Moderate noise:**
```python
# Spectral entropy is better
vad = SpectralEntropyVAD(sample_rate=16000, threshold=0.75)
```

**Real-time with minimal latency:**
```python
# Zero-crossing + energy (no FFT)
vad_zcr = ZeroCrossingVAD(sample_rate=16000)
```

### 2. Frame Parameters

**General speech:**
```python
frame_length = 512   # 32 ms at 16 kHz
hop_length = 256     # 50% overlap
```

**Tradeoffs:**
- Shorter frames: Better time resolution, noisier estimates
- Longer frames: Smoother estimates, delayed detection

### 3. Threshold Tuning

**Conservative (miss less speech):**
```python
threshold = mean_noise + 1.5 * std_noise  # Low threshold
```

**Aggressive (reduce false alarms):**
```python
threshold = mean_noise + 3.0 * std_noise  # High threshold
```

**Adaptive:**
```python
# Update threshold based on recent history
threshold = alpha * threshold_old + (1 - alpha) * threshold_new
```

### 4. Post-Processing

**Hangover (extend speech regions):**
```python
# Keep VAD active for N frames after speech ends
# Prevents choppy detection
hangover_frames = 5  # ~80 ms
```

**Minimum duration:**
```python
# Ignore isolated single-frame detections
min_speech_frames = 3  # ~50 ms
```

**Smoothing:**
```python
# Median filter to remove spurious detections
vad_smooth = median_filter(vad_raw, kernel_size=5)
```

### 5. Multi-Feature Fusion

**Combine multiple cues:**
```python
# Logical AND (conservative)
vad = (energy_vad == 1) AND (entropy_vad == 1)

# Logical OR (sensitive)
vad = (energy_vad == 1) OR (entropy_vad == 1)

# Weighted voting
score = 0.5 * energy_vad + 0.3 * entropy_vad + 0.2 * zcr_vad
vad = (score > threshold)
```

### 6. Noise Estimation

**Initial estimation:**
```python
# Assume first 0.5 seconds are noise-only
noise_frames = frames[0:32]  # 32 frames × 16ms = 512ms
noise_profile = mean(features(noise_frames))
```

**Continuous update:**
```python
# Update during detected silence
if vad == 0:
    noise_profile = alpha * noise_profile + (1-alpha) * current_features
```

### 7. Application-Specific Tuning

**Speech recognition:**
- Bias towards more detection (low threshold)
- Better to process extra frames than miss speech

**Speech coding:**
- Balanced detection
- False positives waste bits, false negatives lose speech

**Comfort noise generation:**
- Accurate silence detection critical
- High threshold to ensure true silence

### 8. Evaluation

**Use standard metrics:**
```python
# Compute precision, recall, F1-score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

**Test on diverse conditions:**
- Multiple SNR levels
- Different noise types
- Various speakers (male, female, children)
- Multiple languages/accents

---

## Advanced Topics

### 1. Machine Learning VAD

Modern VAD often uses:
- Neural networks (LSTM, CNN)
- Trained on large datasets
- Better performance but higher complexity
- Not covered in this library (focus on classical methods)

### 2. Frequency-Domain Features

Beyond entropy:
- Spectral flatness
- Spectral centroid
- Harmonic-to-noise ratio (HNR)

### 3. Model-Based VAD

- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- Statistical models of speech/noise

### 4. Multi-Channel VAD

Using multiple microphones:
- Spatial cues (direction of arrival)
- Coherence between channels
- Beamforming combined with VAD

---

## References

1. Sohn, J., Kim, N. S., & Sung, W. (1999). A statistical model-based voice activity detection. IEEE Signal Processing Letters, 6(1), 1-3.

2. Ramirez, J., Segura, J. C., Benitez, C., De la Torre, A., & Rubio, A. (2004). Efficient voice activity detection algorithms using long-term speech information. Speech communication, 42(3-4), 271-287.

3. Moattar, M. H., & Homayounpour, M. M. (2010). A simple but efficient real-time voice activity detection algorithm. European Signal Processing Conference.

4. Sangwan, A., Chiranth, M. C., Jamadagni, H. S., Sah, R., Prasad, R. V., & Gaurav, V. (2002). VAD techniques for real-time speech transmission on the Internet. IEEE HSNMC.

5. Benyassine, A., Shlomot, E., Su, H. Y., Massaloux, D., Lamblin, C., & Petit, J. P. (1997). ITU-T Recommendation G. 729 Annex B: a silence compression scheme for use with G. 729 optimized for V. 70 digital simultaneous voice and data applications. IEEE Communications magazine, 35(9), 64-73.
