"""Voice Activity Detection (VAD) algorithms."""

from speechalgo.vad.energy_based import EnergyBasedVAD
from speechalgo.vad.spectral_entropy import SpectralEntropyVAD
from speechalgo.vad.zero_crossing import ZeroCrossingVAD

__all__ = [
    "EnergyBasedVAD",
    "SpectralEntropyVAD",
    "ZeroCrossingVAD",
]
