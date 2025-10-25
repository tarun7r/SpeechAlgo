# Speech Enhancement Theory

This document covers the theoretical foundations of noise reduction algorithms implemented in SpeechAlgo.

## Table of Contents

1. [Enhancement Problem Formulation](#enhancement-problem-formulation)
2. [Spectral Subtraction](#spectral-subtraction)
3. [Wiener Filtering](#wiener-filtering)
4. [Noise Gate](#noise-gate)
5. [Performance Comparison](#performance-comparison)
6. [Best Practices](#best-practices)

---

## Enhancement Problem Formulation

### Signal Model

Noisy speech:

```
y(t) = x(t) + n(t)
```

where:
- y(t): Observed noisy signal
- x(t): Clean speech (unknown, to be estimated)
- n(t): Additive noise (unknown, to be estimated)

**Assumptions:**
1. Noise and speech are uncorrelated
2. Noise is additive (not multiplicative or convolutional)
3. Noise statistics can be estimated

### Frequency Domain

```
Y(ω) = X(ω) + N(ω)
```

Power spectral density:

```
|Y(ω)|² = |X(ω)|² + |N(ω)|²  (for uncorrelated signals)
```

### Goals

1. **Primary:** Estimate clean speech X̂(ω) from Y(ω)
2. **Minimize:** Mean Square Error E[|X(ω) - X̂(ω)|²]
3. **Balance:** Noise reduction vs. speech distortion

### Performance Metrics

**Signal-to-Noise Ratio (SNR):**

```
SNR = 10 * log10( P_signal / P_noise )
```

**SNR Improvement:**

```
ΔSNR = SNR_output - SNR_input
```

**Speech Distortion:**

```
SD = distance(X̂, X)  (spectral distance, PESQ, etc.)
```

**Perceptual Quality:**
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- MOS (Mean Opinion Score)

---

## Spectral Subtraction

### Principle

Subtract estimated noise spectrum from noisy speech spectrum in the frequency domain.

### Basic Algorithm

1. **Estimate noise spectrum:**
   ```
   |N̂(ω)|² = average of |Y(ω)|² during silence periods
   ```

2. **Subtract in power domain:**
   ```
   |X̂(ω)|² = |Y(ω)|² - |N̂(ω)|²
   ```

3. **Half-wave rectification:**
   ```
   |X̂(ω)|² = max(|X̂(ω)|², floor)
   ```

4. **Reconstruct time domain:**
   ```
   X̂(ω) = sqrt(|X̂(ω)|²) * exp(j * phase(Y(ω)))
   x̂(t) = ISTFT(X̂(ω))
   ```

### Mathematical Derivation

From the power spectrum relationship:

```
|X(ω)|² = |Y(ω)|² - |N(ω)|²
```

Taking square root for magnitude:

```
|X̂(ω)| = sqrt(|Y(ω)|² - |N̂(ω)|²)
```

### Over-Subtraction

To account for estimation errors and improve noise reduction:

```
|X̂(ω)|² = |Y(ω)|² - α * |N̂(ω)|²
```

where α > 1 is the over-subtraction factor (typical: 1.5-4.0).

**Trade-off:**
- α too low: Residual noise remains (musical noise)
- α too high: Speech distortion increases

### Spectral Floor

To prevent negative or zero values:

```
|X̂(ω)|² = max(|Y(ω)|² - α*|N̂(ω)|², β*|Y(ω)|²)
```

where β is the spectral floor (typical: 0.01-0.1).

### Musical Noise

**Problem:** Spectral subtraction creates "musical noise" - isolated spectral peaks that sound like random tones.

**Cause:** Random fluctuations in noise estimate create sporadic positive residuals.

**Solutions:**

1. **Over-subtraction + floor:** Aggressive removal with safety net
2. **Spectral smoothing:** Average neighboring frequency bins
3. **Temporal smoothing:** Average consecutive frames
4. **Soft thresholding:** Gradual attenuation instead of hard subtraction

### Advantages

✅ Simple and intuitive  
✅ Fast computation (one STFT)  
✅ Aggressive noise reduction possible  
✅ Works well for stationary noise

### Limitations

❌ Musical noise artifacts  
❌ Speech distortion at low SNR  
❌ Assumes stationary noise  
❌ Requires reliable noise estimate

### When to Use

- Stationary background noise (fan, hum, white noise)
- When aggressive reduction is needed
- Real-time applications (low latency)
- Preliminary enhancement before other processing

---

## Wiener Filtering

### Principle

Wiener filtering is the optimal linear filter in the minimum mean square error (MMSE) sense.

### Optimal Gain

The Wiener filter gain is derived by minimizing:

```
E[ |X(ω) - G(ω)*Y(ω)|² ]
```

Solution:

```
G(ω) = P_X(ω) / (P_X(ω) + P_N(ω))
```

where:
- P_X(ω): Clean speech power spectrum
- P_N(ω): Noise power spectrum

### Alternative Form

Using SNR = P_X / P_N:

```
G(ω) = SNR(ω) / (1 + SNR(ω))
```

Or equivalently:

```
G(ω) = P_Y(ω) - P_N(ω) / P_Y(ω) = 1 - P_N(ω)/P_Y(ω)
```

### A Priori vs. A Posteriori SNR

**A posteriori SNR** (observed):

```
γ(ω) = |Y(ω)|² / P_N(ω)
```

**A priori SNR** (estimated):

```
ξ(ω) = P_X(ω) / P_N(ω)
```

**Wiener gain using a priori SNR:**

```
G(ω) = ξ(ω) / (1 + ξ(ω))
```

### Decision-Directed Approach

To reduce musical noise, use temporal smoothing:

```
ξ̂[t, ω] = α * |X̂[t-1, ω]|²/P_N(ω) + (1-α) * max(γ[t, ω] - 1, 0)
```

where α ≈ 0.98 is the smoothing factor.

This balances:
- Past estimate (slow adaptation, smooth)
- Current observation (fast adaptation, noisy)

### Noise Floor

To prevent over-suppression:

```
G_final(ω) = max(G(ω), G_min)
```

where G_min is the minimum gain (typical: 0.01-0.1).

**Purpose:**
- Maintains some signal energy
- Prevents "holes" in spectrum
- Preserves intelligibility

### Algorithm Steps

1. **Estimate noise spectrum P_N(ω)**
   ```
   From initial silence or continuous update
   ```

2. **Compute power spectrum:**
   ```
   P_Y(ω) = |Y(ω)|²
   ```

3. **Estimate clean speech spectrum:**
   ```
   P_X(ω) = P_Y(ω) - P_N(ω)
   ```

4. **Compute Wiener gain:**
   ```
   G(ω) = P_X(ω) / P_Y(ω)
   ```

5. **Apply floor:**
   ```
   G(ω) = max(G(ω), G_min)
   ```

6. **Apply gain:**
   ```
   X̂(ω) = G(ω) * Y(ω)
   ```

7. **Reconstruct time domain:**
   ```
   x̂(t) = ISTFT(X̂(ω))
   ```

### Advantages

✅ Optimal in MMSE sense  
✅ Less musical noise than spectral subtraction  
✅ Smooth gain function  
✅ Theoretically grounded  
✅ Good balance of noise reduction and distortion

### Limitations

❌ Requires noise spectrum estimate  
❌ Assumes Gaussian statistics  
❌ Degrades with non-stationary noise  
❌ Can cause speech distortion at low SNR  
❌ Slower than spectral subtraction (iterative)

### When to Use

- General purpose noise reduction
- When smooth enhancement is needed
- Stationary or slowly-varying noise
- When speech quality is important

---

## Noise Gate

### Principle

A noise gate is a simple threshold-based amplitude processor:

```
If level < threshold: Attenuate (gate closed)
If level ≥ threshold: Pass through (gate open)
```

### Parameters

**Threshold:** Level below which signal is attenuated (dB)

**Attack time:** How quickly gate opens when signal exceeds threshold

**Release time:** How quickly gate closes when signal falls below threshold

**Ratio/Range:** Amount of attenuation when gate is closed

### Algorithm

```
if level[t] < threshold:
    gain[t] = approach(0, attack_time)  # Close gate
else:
    gain[t] = approach(1, release_time)  # Open gate

output[t] = input[t] * gain[t]
```

### Smooth Transitions

Instead of hard switching, use exponential smoothing:

```
if level[t] < threshold:
    target = min_gain
    time_constant = release_time
else:
    target = 1.0
    time_constant = attack_time

gain[t] = α * gain[t-1] + (1-α) * target
```

where α = exp(-1 / (time_constant * sample_rate)).

### RMS Level Estimation

```
level[t] = sqrt( (1/N) * sum_{n} x[t*hop + n]² )
```

Often computed in dB:

```
level_dB[t] = 20 * log10(level[t] + ε)
```

### Advantages

✅ Very simple, minimal computation  
✅ No frequency domain processing  
✅ No musical noise  
✅ Preserves speech spectrum  
✅ Real-time with minimal latency

### Limitations

❌ Cannot remove noise during speech  
❌ Only works on silence/pause regions  
❌ Abrupt transitions can be audible  
❌ Threshold requires tuning  
❌ Not effective for high noise levels

### When to Use

- As preprocessing (remove silence)
- Clean environments with silence gaps
- Complementary to spectral methods
- Real-time with strict latency requirements

---

## Performance Comparison

### Noise Reduction

| Method | Stationary | Non-stationary | Low SNR | High SNR |
|--------|-----------|----------------|---------|----------|
| **Spectral Subtraction** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Wiener Filter** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Noise Gate** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |

### Speech Quality

| Method | Distortion | Musical Noise | Naturalness |
|--------|-----------|---------------|-------------|
| **Spectral Subtraction** | Moderate | High | Moderate |
| **Wiener Filter** | Low | Low | Good |
| **Noise Gate** | None | None | Excellent |

### Computational Cost

| Method | Complexity | Latency | Memory |
|--------|-----------|---------|---------|
| **Spectral Subtraction** | O(N log N) | Low | Moderate |
| **Wiener Filter** | O(N log N) | Moderate | Moderate |
| **Noise Gate** | O(N) | Minimal | Minimal |

---

## Best Practices

### 1. Noise Estimation

**Initial estimation:**
```python
# Use first 0.5-1.0 seconds as noise-only
noise_frames = signal[0:sample_rate]
noise_spectrum = estimate_spectrum(noise_frames)
```

**Continuous update:**
```python
# Update during VAD-detected silence
if vad == 0:  # Silence/noise only
    noise_spectrum = α * noise_spectrum + (1-α) * current_spectrum
```

### 2. Parameter Tuning

**Spectral subtraction:**
```python
over_subtraction = 2.0    # Aggressive: 3-4, Conservative: 1.5-2
spectral_floor = 0.02     # 2% of original
smoothing_alpha = 0.7     # Temporal smoothing
```

**Wiener filter:**
```python
min_gain = 0.01           # -40 dB suppression
smoothing_alpha = 0.98    # Decision-directed
noise_floor = 1e-10       # Prevent division by zero
```

**Noise gate:**
```python
threshold_db = -40        # Relative to peak
attack_ms = 10            # Fast opening
release_ms = 100          # Slow closing
min_gain = 0.01           # -40 dB when closed
```

### 3. Combination Strategies

**Serial (cascade):**
```python
# Noise gate first (remove silence)
signal = noise_gate(signal)
# Then spectral method (reduce in-speech noise)
signal = wiener_filter(signal)
```

**Parallel (fusion):**
```python
# Apply both methods
enhanced1 = spectral_subtraction(signal)
enhanced2 = wiener_filter(signal)
# Combine with weights
enhanced = α * enhanced1 + (1-α) * enhanced2
```

### 4. Post-Processing

**Smoothing:**
```python
# Temporal smoothing of gain
gain_smooth[t] = α * gain_smooth[t-1] + (1-α) * gain[t]
```

**Limiter:**
```python
# Prevent amplification beyond original
enhanced = min(enhanced, noisy)
```

### 5. Real-Time Considerations

**Frame size vs. latency:**
```python
frame_size = 512    # 32 ms at 16 kHz
hop_size = 256      # 50% overlap, 16 ms latency
```

**Look-ahead:**
- None needed for spectral subtraction/Wiener
- Can improve noise gate (detect attack earlier)

### 6. Quality Assessment

**Objective metrics:**
```python
snr_improvement = snr(enhanced) - snr(noisy)
spectral_distance = distance(enhanced_spec, clean_spec)
```

**Subjective testing:**
- Listen tests (MOS scoring)
- A/B comparison
- Test on target application (ASR, telephony, etc.)

---

## References

1. Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral subtraction. IEEE TASSP, 27(2), 113-120.

2. Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator. IEEE TASSP, 32(6), 1109-1121.

3. Scalart, P., & Filho, J. V. (1996). Speech enhancement based on a priori signal to noise estimation. ICASSP.

4. Loizou, P. C. (2013). Speech enhancement: theory and practice. CRC press.

5. Vary, P., & Martin, R. (2006). Digital speech transmission: Enhancement, coding and error concealment. John Wiley & Sons.
