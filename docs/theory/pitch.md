# Pitch Detection Theory

This document covers the theoretical foundations of pitch detection algorithms implemented in SpeechAlgo.

## Table of Contents

1. [Pitch Detection Fundamentals](#pitch-detection-fundamentals)
2. [Autocorrelation Method](#autocorrelation-method)
3. [YIN Algorithm](#yin-algorithm)
4. [Cepstral Pitch Detection](#cepstral-pitch-detection)
5. [Harmonic Product Spectrum (HPS)](#harmonic-product-spectrum-hps)
6. [Algorithm Comparison](#algorithm-comparison)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Pitch Detection Fundamentals

### Definition

**Pitch (F0):** The fundamental frequency of a periodic sound, perceived as its musical note.

For speech:
- Male: 85-180 Hz
- Female: 165-255 Hz  
- Children: 250-400 Hz

### Mathematical Model

Voiced speech can be modeled as:

```
s(t) = e(t) * h(t)
```

where:
- e(t): Periodic excitation (glottal pulses) at frequency F0
- h(t): Vocal tract filter (formants)
- *: Convolution

**Goal:** Extract F0 from s(t) despite interference from h(t).

### Challenges

1. **Formants:** Vocal tract resonances can mask F0
2. **Octave errors:** Detecting 2×F0 or F0/2 instead of F0
3. **Harmonic interference:** Multiple harmonics complicate detection
4. **Voiced/unvoiced:** Only voiced speech has pitch
5. **Noisy conditions:** Background noise obscures periodicity

### Evaluation Metrics

**Gross Pitch Error (GPE):**
```
GPE = |F0_estimated - F0_true| / F0_true > threshold (typically 20%)
```

**Fine Pitch Error (FPE):**
```
FPE = |F0_estimated - F0_true| / F0_true (for correct detections)
```

**Voicing Decision Error (VDE):**
- False positive: Detecting pitch in unvoiced frame
- False negative: Missing pitch in voiced frame

---

## Autocorrelation Method

### Principle

Periodic signals correlate well with delayed versions of themselves. The delay at which maximum correlation occurs corresponds to the pitch period.

### Autocorrelation Function

```
R(τ) = sum_{n} x[n] * x[n + τ]
```

where τ is the lag (delay in samples).

### Algorithm Steps

1. **Compute autocorrelation:**
   ```
   R(τ) for τ = 0, 1, 2, ..., max_lag
   ```

2. **Find first peak after τ=0:**
   ```
   Search R(τ) for τ_min ≤ τ ≤ τ_max
   ```

3. **Convert lag to frequency:**
   ```
   F0 = sample_rate / τ_peak
   ```

### Lag Range

For F0 range [F0_min, F0_max]:

```
τ_min = sample_rate / F0_max
τ_max = sample_rate / F0_min
```

Example at 16 kHz for 80-400 Hz:
```
τ_min = 16000 / 400 = 40 samples
τ_max = 16000 / 80 = 200 samples
```

### Advantages

✅ Simple and intuitive  
✅ Works well for clean, periodic signals  
✅ Fast computation (can use FFT)  
✅ Well-understood mathematically

### Limitations

❌ Sensitive to formants (peaks at formant periods)  
❌ Can produce octave errors  
❌ Performance degrades in noise  
❌ Requires threshold tuning

### Improvements

**Center clipping:**
```
Remove samples with |x[n]| < threshold
Reduces formant influence
```

**AMDF (Average Magnitude Difference Function):**
```
D(τ) = sum_{n} |x[n] - x[n+τ]|
Find minimum instead of maximum
```

---

## YIN Algorithm

### Motivation

YIN improves upon autocorrelation by addressing its main weaknesses:
- Formant sensitivity
- Lack of aperiodicity measure
- Threshold dependency

### Key Innovations

1. **Difference function instead of correlation**
2. **Cumulative mean normalized difference**
3. **Absolute threshold on confidence**
4. **Parabolic interpolation for sub-sample accuracy**

### Algorithm Steps

#### Step 1: Difference Function

```
d(τ) = sum_{n=0}^{W-1} (x[n] - x[n+τ])^2
```

where W is window size.

Similar to AMDF but squared differences.

#### Step 2: Cumulative Mean Normalized Difference

```
d'(τ) = d(τ) / [ (1/τ) * sum_{j=1}^{τ} d(j) ]
```

with d'(0) = 1 by definition.

**Effect:**
- Normalizes by average over all previous lags
- Makes threshold independent of signal amplitude
- Reduces bias towards longer periods

#### Step 3: Absolute Threshold

Find the smallest τ where:

```
d'(τ) < threshold (typically 0.1-0.15)
```

within the valid F0 range.

**Advantage:** Measures confidence in periodicity, not just correlation strength.

#### Step 4: Parabolic Interpolation

Refine τ estimate using neighbors:

```
τ_refined = τ + (d'[τ-1] - d'[τ+1]) / (2 * (d'[τ-1] - 2*d'[τ] + d'[τ+1]))
```

Provides sub-sample accuracy.

### Advantages over Autocorrelation

✅ Better formant rejection  
✅ Built-in confidence measure  
✅ More robust to amplitude variations  
✅ Fewer octave errors  
✅ Sub-sample accuracy

### Computational Complexity

O(N × M) where:
- N: frame length
- M: number of lags to search

**Optimization:** Can use FFT-based computation for O(N log N).

### Parameter Selection

| Parameter | Typical Value | Effect |
|-----------|--------------|---------|
| Window | 2048 samples (~128ms) | Longer → better for low pitch |
| Threshold | 0.1 | Lower → more sensitive |
| F0 range | 80-400 Hz | Match expected pitch range |

---

## Cepstral Pitch Detection

### Principle

The cepstrum separates the excitation (pitch) from the vocal tract filter (formants) using logarithmic transformation.

### Cepstrum Definition

The **real cepstrum** is:

```
c[n] = IFFT(log(|FFT(x[n])|))
```

The name "cepstrum" is an anagram of "spectrum", and the independent variable is called "quefrency" (anagram of "frequency").

### Why It Works

In the source-filter model:

```
Spectrum: S(f) = E(f) × H(f)
Log spectrum: log(S(f)) = log(E(f)) + log(H(f))
Cepstrum: IFFT converts multiplication to addition
```

Result:
- **Low quefrencies:** Smooth spectral envelope (vocal tract)
- **High quefrencies:** Fine structure (pitch harmonics)

### Algorithm Steps

1. **Compute power spectrum:**
   ```
   S[f] = |FFT(x[n])|^2
   ```

2. **Take logarithm:**
   ```
   log_S[f] = log(S[f] + ε)
   ```

3. **Compute cepstrum:**
   ```
   c[q] = IFFT(log_S[f])
   ```

4. **Apply liftering:**
   ```
   c_liftered[q] = c[q] for q > lifter_cutoff
   c_liftered[q] = 0 for q ≤ lifter_cutoff
   ```

5. **Find peak in quefrency range:**
   ```
   q_peak = argmax(c_liftered[q]) for q_min ≤ q ≤ q_max
   ```

6. **Convert to frequency:**
   ```
   F0 = sample_rate / q_peak
   ```

### Quefrency Range

```
q_min = sample_rate / F0_max
q_max = sample_rate / F0_min
```

### Liftering

Liftering is filtering in the quefrency domain (anagram of "filtering"):

**Purpose:** Remove low quefrency components that represent the vocal tract, leaving only the excitation periodicity.

**Typical cutoff:** 1-5 ms (16-80 samples at 16 kHz)

### Advantages

✅ Excellent formant rejection  
✅ Separates source from filter  
✅ Theoretically grounded  
✅ Robust to spectral tilt

### Limitations

❌ Requires longer frames (512-1024 samples)  
❌ Two FFT operations (slower)  
❌ Sensitive to spectral zeros  
❌ Log can amplify noise

### When to Use

- Speech with strong formants (vowels)
- When formants interfere with other methods
- Offline processing (not real-time critical)

---

## Harmonic Product Spectrum (HPS)

### Principle

Voiced speech has harmonics at integer multiples of F0:

```
Harmonics: F0, 2×F0, 3×F0, 4×F0, ...
```

HPS downsamples the spectrum by factors 2, 3, 4, ... and multiplies them. The product has a strong peak at F0 where all harmonics align.

### Algorithm Steps

1. **Compute magnitude spectrum:**
   ```
   S[k] = |FFT(x[n])|
   ```

2. **Downsample spectrum:**
   ```
   S_2[k] = S[2k]  (every 2nd bin)
   S_3[k] = S[3k]  (every 3rd bin)
   S_4[k] = S[4k]  (every 4th bin)
   ```

3. **Compute harmonic product:**
   ```
   HPS[k] = S[k] × S_2[k] × S_3[k] × S_4[k] × ...
   ```

4. **Find peak:**
   ```
   k_peak = argmax(HPS[k]) for k_min ≤ k ≤ k_max
   ```

5. **Convert to frequency:**
   ```
   F0 = k_peak × (sample_rate / N_fft)
   ```

### Example

For F0 = 100 Hz:
```
Original spectrum: peaks at 100, 200, 300, 400 Hz
Downsample ÷2:    peaks at 50, 100, 150, 200 Hz
Downsample ÷3:    peaks at 33, 67, 100, 133 Hz
Product:          strongest peak at 100 Hz
```

### Number of Harmonics

Typical: 3-7 harmonics

**Trade-off:**
- More harmonics → Better discrimination but slower
- Fewer harmonics → Faster but less robust

### Advantages

✅ Robust to missing fundamental  
✅ Works well with clear harmonics  
✅ Frequency-domain (no lag search)  
✅ Good for octave error reduction

### Limitations

❌ Requires harmonic structure  
❌ Fails with inharmonic sounds  
❌ Sensitive to noise between harmonics  
❌ Computationally expensive

### When to Use

- Strong harmonic signals (vowels, music)
- When fundamental is weak or missing
- Offline processing with clear signal

---

## Algorithm Comparison

### Accuracy

| Algorithm | Clean | Moderate Noise | Low SNR |
|-----------|-------|----------------|---------|
| Autocorrelation | Good | Moderate | Poor |
| YIN | Excellent | Good | Moderate |
| Cepstral | Excellent | Good | Moderate |
| HPS | Excellent | Good | Poor |

### Speed

| Algorithm | Complexity | Real-time Capable |
|-----------|------------|-------------------|
| Autocorrelation | O(N log N) | Yes |
| YIN | O(N×M) | Yes (with optimization) |
| Cepstral | O(N log N) | Yes (two FFTs) |
| HPS | O(N log N) | Yes |

### Robustness

| Algorithm | Formants | Octave Errors | Missing F0 |
|-----------|----------|---------------|------------|
| Autocorrelation | Poor | Common | Poor |
| YIN | Good | Rare | Moderate |
| Cepstral | Excellent | Rare | Moderate |
| HPS | Good | Very Rare | Excellent |

### Recommended Usage

```
Clean speech, real-time:       Autocorrelation or YIN
Formant-heavy speech:          Cepstral or YIN
Harmonic sounds:               HPS
General purpose:               YIN (best balance)
```

---

## Common Issues and Solutions

### 1. Octave Errors

**Problem:** Detecting 2×F0 or F0/2 instead of F0

**Solutions:**
- Use YIN (built-in octave error reduction)
- Check harmonics (HPS)
- Post-process with median filter
- Enforce F0 continuity (pitch tracking)

### 2. Voiced/Unvoiced Confusion

**Problem:** Detecting pitch in unvoiced segments

**Solutions:**
- Combine with VAD (check energy, ZCR)
- Use confidence threshold (YIN's d'(τ))
- Check periodicity strength
- Require minimum correlation/confidence

### 3. Low SNR Performance

**Problem:** Noise obscures periodicity

**Solutions:**
- Enhance signal first (spectral subtraction, Wiener)
- Use longer frames (more periods)
- Increase confidence threshold
- Use multiple frames (median/mode)

### 4. Pitch Doubling/Halving

**Problem:** Sporadic jumps to 2×F0 or F0/2

**Solutions:**
```python
# Median filtering
pitch_smooth = median_filter(pitch_raw, kernel_size=5)

# Continuity constraint
if abs(pitch[t] - pitch[t-1]) > threshold:
    pitch[t] = pitch[t-1]  # Keep previous value
```

### 5. Frame Length Selection

**Too short:** May not contain full period
**Too long:** Loses time resolution

**Recommendation:**
```
frame_length = 4 * sample_rate / F0_min
```

Example: For F0_min = 80 Hz at 16 kHz:
```
frame_length = 4 * 16000 / 80 = 800 samples ≈ 50ms
```

### 6. Confidence Thresholds

**Autocorrelation:**
```python
if max_correlation < 0.5:  # Weak periodicity
    F0 = 0  # Unvoiced
```

**YIN:**
```python
if min(d_prime) > threshold:  # No strong dip
    F0 = 0  # Unvoiced
```

---

## References

1. Rabiner, L. R. (1977). On the use of autocorrelation analysis for pitch detection. IEEE TASSP, 25(1), 24-33.

2. De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. JASA, 111(4), 1917-1930.

3. Noll, A. M. (1967). Cepstrum pitch determination. JASA, 41(2), 293-309.

4. Schroeder, M. R. (1968). Period histogram and product spectrum: New methods for fundamental-frequency measurement. JASA, 43(4), 829-834.

5. Huang, F., & Lee, T. (2012). Pitch estimation in noisy speech based on temporal accumulation of spectrum peaks. ICASSP.
