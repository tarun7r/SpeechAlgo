"""
Delta and delta-delta features.

Delta features (also called velocity or differential features) capture
the temporal dynamics of static features. They are essential for improving
the performance of speech recognition and other time-series tasks.

Mathematical Foundation
-----------------------
Delta features compute the first-order time derivative of features:

    Δ[t] = Σ_{n=-N}^{N} n * c[t+n] / Σ_{n=-N}^{N} n²

where:
- c[t] is the feature value at time t
- N is the delta window (typically 2-5 frames)
- n is the time offset

This is essentially a regression slope computed over a local window.

Delta-delta (acceleration) features are second-order derivatives:

    ΔΔ[t] = Δ(Δ[t])

For speech recognition, a typical feature vector includes:
- Static features (e.g., 13 MFCCs)
- Delta features (13 Δ-MFCCs)
- Delta-delta features (13 ΔΔ-MFCCs)
Total: 39 features per frame

References
----------
.. [1] Furui, S. (1986). "Speaker-independent isolated word recognition using
       dynamic features of speech spectrum". IEEE Transactions on ASSP, 34(1).
.. [2] Rabiner, L., & Juang, B. H. (1993). "Fundamentals of speech recognition".
.. [3] Young, S., et al. (2006). "The HTK book" (HTK Version 3.4). Cambridge
       University Engineering Department.
"""

import numpy as np
import numpy.typing as npt


class DeltaFeatures:
    """
    Compute delta and delta-delta features.

    Delta features capture temporal dynamics by computing derivatives
    of static features over time.

    Parameters
    ----------
    width : int, default=2
        Width of the delta window in frames (N in formula).
        Typical values: 2-5 frames

    Examples
    --------
    >>> # Compute deltas for MFCC features
    >>> from speechalgo.preprocessing import MFCC
    >>> mfcc_extractor = MFCC(sample_rate=16000, n_mfcc=13)
    >>> mfcc = mfcc_extractor.process(audio)
    >>> print(f"MFCC shape: {mfcc.shape}")  # (13, n_frames)

    >>> delta = DeltaFeatures(width=2)
    >>> mfcc_delta = delta.compute_delta(mfcc)
    >>> mfcc_delta2 = delta.compute_delta(mfcc_delta)
    >>> print(f"Delta shape: {mfcc_delta.shape}")
    >>> print(f"Delta-delta shape: {mfcc_delta2.shape}")

    >>> # Combine into full feature vector
    >>> features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    >>> print(f"Full feature shape: {features.shape}")  # (39, n_frames)

    >>> # Visualize deltas
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    >>> im1 = axes[0].imshow(mfcc, aspect='auto', origin='lower', cmap='coolwarm')
    >>> axes[0].set_title('Static MFCCs')
    >>> axes[0].set_ylabel('Coefficient')
    >>> im2 = axes[1].imshow(mfcc_delta, aspect='auto', origin='lower', cmap='coolwarm')
    >>> axes[1].set_title('Delta MFCCs')
    >>> axes[1].set_ylabel('Coefficient')
    >>> im3 = axes[2].imshow(mfcc_delta2, aspect='auto', origin='lower', cmap='coolwarm')
    >>> axes[2].set_title('Delta-Delta MFCCs')
    >>> axes[2].set_ylabel('Coefficient')
    >>> axes[2].set_xlabel('Frame')
    >>> plt.colorbar(im1, ax=axes[0])
    >>> plt.colorbar(im2, ax=axes[1])
    >>> plt.colorbar(im3, ax=axes[2])
    >>> plt.tight_layout()
    >>> plt.show()

    Notes
    -----
    Delta features capture:
    - Temporal dynamics and trajectories
    - Velocity of feature changes
    - Transition information
    - Coarticulation effects in speech

    Usage recommendations:
    - width = 2: Fast changes, good time resolution
    - width = 5: Smooth changes, robust to noise
    - width = 2 common for speech recognition
    - width = 3-5 common for speaker recognition

    Boundary handling:
    - Features at boundaries are padded by replication
    - First/last `width` frames may have reduced accuracy

    Implementation note:
    - Uses efficient vectorized computation
    - Normalized by sum of squared offsets
    - Handles any feature dimensionality (MFCCs, spectral, etc.)
    """

    def __init__(self, width: int = 2):
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.width = width

        # Precompute normalization factor
        # denominator = Σ_{n=-width}^{width} n²
        self.normalizer = sum(n**2 for n in range(-width, width + 1))

    def compute_delta(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute delta (first derivative) features.

        Parameters
        ----------
        features : ndarray, shape (n_features, n_frames)
            Input feature matrix (e.g., MFCCs)

        Returns
        -------
        delta : ndarray, shape (n_features, n_frames)
            Delta features

        Examples
        --------
        >>> delta = DeltaFeatures(width=2)
        >>> mfcc_delta = delta.compute_delta(mfcc)
        >>> # Delta features have same shape as input
        >>> assert mfcc_delta.shape == mfcc.shape
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D (n_features, n_frames), got shape {features.shape}"
            )

        n_features, n_frames = features.shape

        # Pad features at boundaries
        padded = self._pad_features(features)

        # Compute delta using convolution-like operation
        delta = np.zeros_like(features)

        for t in range(n_frames):
            # Extract window: [t+width-width : t+width+width+1]
            # In padded array: [t : t+2*width+1]
            window = padded[:, t : t + 2 * self.width + 1]

            # Compute weighted sum: Σ n * c[t+n]
            for n in range(-self.width, self.width + 1):
                idx = n + self.width
                delta[:, t] += n * window[:, idx]

            # Normalize
            delta[:, t] /= self.normalizer

        return delta.astype(np.float32)

    def compute_delta_delta(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Compute delta-delta (second derivative) features.

        This is equivalent to computing delta of delta features,
        representing acceleration.

        Parameters
        ----------
        features : ndarray, shape (n_features, n_frames)
            Input feature matrix (e.g., MFCCs)

        Returns
        -------
        delta_delta : ndarray, shape (n_features, n_frames)
            Delta-delta features

        Examples
        --------
        >>> delta = DeltaFeatures(width=2)
        >>> mfcc_delta2 = delta.compute_delta_delta(mfcc)
        >>> # Or equivalently:
        >>> mfcc_delta = delta.compute_delta(mfcc)
        >>> mfcc_delta2_alt = delta.compute_delta(mfcc_delta)
        >>> np.allclose(mfcc_delta2, mfcc_delta2_alt)
        True
        """
        # Delta-delta is just delta of delta
        delta = self.compute_delta(features)
        delta_delta = self.compute_delta(delta)

        return delta_delta

    def _pad_features(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Pad features at boundaries for delta computation.

        Uses edge replication (repeat first/last frame).

        Parameters
        ----------
        features : ndarray, shape (n_features, n_frames)
            Input features

        Returns
        -------
        padded : ndarray, shape (n_features, n_frames + 2*width)
            Padded features
        """
        # Pad by replicating edge frames
        padded = np.pad(
            features,
            ((0, 0), (self.width, self.width)),
            mode="edge",
        )

        return padded

    def __repr__(self) -> str:
        return f"DeltaFeatures(width={self.width})"
