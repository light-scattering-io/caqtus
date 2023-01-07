from collections import defaultdict
from collections.abc import Sequence
from itertools import product, chain
from typing import Iterable

import numpy as np
from numba import njit, float64, prange
from pydantic import validator, Field
from scipy.optimize import basinhopping

from settings_model import SettingsModel


class StaticTrapGenerator(SettingsModel):
    frequencies: Sequence[float] = Field(allow_mutation=False)
    amplitudes: Sequence[float] = Field()
    phases: Sequence[float] = Field()
    sampling_rate: float = Field(gt=0, allow_mutation=False)
    number_samples: int = Field(gt=0, allow_mutation=False)

    @validator("frequencies", pre=True)
    def validate_frequencies(cls, frequencies):
        return tuple(frequencies)

    @validator("amplitudes", pre=True)
    def validate_amplitudes(cls, amplitudes, values):
        frequencies = values["frequencies"]
        if len(amplitudes) != len(frequencies):
            raise ValueError(
                "Number of amplitudes must be the same than the number of frequencies"
            )
        return tuple(amplitudes)

    @validator("phases", pre=True)
    def validate_phases(cls, phases, values):
        frequencies = values["frequencies"]
        if len(phases) != len(frequencies):
            raise ValueError(
                "Number of phases must be the same than the number of frequencies"
            )
        return tuple(phases)

    def compute_signal(self) -> np.ndarray["number_samples", np.float32]:
        return compute_signal_numba(
            times=np.array(self.times, dtype=np.float64),
            amplitudes=np.array(self.amplitudes, dtype=np.float64),
            frequencies=np.array(self.frequencies, dtype=np.float64),
            phases=np.array(self.phases, dtype=np.float64),
        )

    @property
    def times(self):
        return np.arange(self.number_samples) / self.sampling_rate

    def optimize_phases(self):
        """Changes its phases to reduce the peak values of the signal envelope."""

        segment_frequency = self.sampling_rate / self.number_samples
        optimal_phases = compute_optimized_phases(
            self.frequencies, self.amplitudes, segment_frequency
        )
        self.phases = optimal_phases


@njit(parallel=True)
def compute_signal_numba(
    times: float64[:],
    amplitudes: float64[:],
    frequencies: float64[:],
    phases: float64[:],
) -> float64[:]:
    result = np.zeros_like(times)
    t = times
    number_tones = len(amplitudes)
    for tone in prange(number_tones):
        amplitude = amplitudes[tone]
        frequency = frequencies[tone]
        phase = phases[tone]

        result += amplitude * np.sin(2 * np.pi * t * frequency + phase)
    return result


def compute_optimized_phases(
    frequencies: Sequence[float],
    amplitudes: Sequence[float],
    segment_frequency: float,
    initial_phases: Sequence[float] = None,
):
    amplitudes = np.array(amplitudes)
    frequencies = np.array(frequencies)

    integer_frequencies = np.round(frequencies / segment_frequency).astype(int)
    indexes = np.array(list(_compute_quadruplets(integer_frequencies)))

    def to_minimize(phases):
        a = (amplitudes * np.exp(1j * phases))[indexes]
        return np.real(np.sum(a[:, 0] * np.conj(a[:, 1]) * np.conj(a[:, 2]) * a[:, 3]))

    if initial_phases:
        initial_phases = np.array(initial_phases)
    else:
        initial_phases = np.random.uniform(0, 2 * np.pi, len(frequencies))

    solution = basinhopping(to_minimize, initial_phases)
    return solution.x


def _compute_quadruplets(
    frequencies: Sequence[int],
) -> Iterable[tuple[int, int, int, int]]:
    """Computes all indexes i,j,k,l such that frequencies[i]-frequencies[j] == frequencies[k]-frequencies[l]"""

    diffs = defaultdict(list)
    for (i, fi), (j, fj) in product(enumerate(frequencies), repeat=2):
        diffs[fi - fj].append((i, j))

    indexes = chain(
        *map(lambda y: map(lambda x: x[0] + x[1], product(y, repeat=2)), diffs.values())
    )
    return indexes
