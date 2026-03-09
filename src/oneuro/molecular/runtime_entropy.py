"""Runtime entropy controller for ongoing stochastic network dynamics.

The controller is deliberately generic: it accepts a callback that yields
fresh 64-bit entropy words and turns those into periodic reseeds of a local
NumPy PCG64 generator. Experiments can back the callback with QCicada, a
software control, or any other source without coupling the core simulator to
experiment-only code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(slots=True, frozen=True)
class RuntimeEntropyBlock:
    """One runtime entropy refresh event."""

    source: str
    seed64: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "seed64": int(self.seed64),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class RuntimeEntropySettings:
    """How runtime entropy is injected into ongoing dynamics."""

    current_noise_std: float = 0.0
    microtubule_collapse_jitter_std: float = 0.0

    def as_metadata_record(self) -> dict[str, Any]:
        return {
            "current_noise_std": float(self.current_noise_std),
            "microtubule_collapse_jitter_std": float(
                self.microtubule_collapse_jitter_std
            ),
        }


RuntimeEntropySupplier = Callable[[int, int], RuntimeEntropyBlock]


def _mix_seed(
    base_seed: int,
    entropy_seed: int,
    refresh_index: int,
    step_count: int,
) -> int:
    """Deterministically mix base seed, runtime entropy, and schedule state."""
    payload = (
        int(base_seed).to_bytes(8, "big", signed=False)
        + int(entropy_seed & ((1 << 64) - 1)).to_bytes(8, "big", signed=False)
        + int(refresh_index).to_bytes(8, "big", signed=False)
        + int(step_count).to_bytes(8, "big", signed=False)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@dataclass(slots=True)
class RuntimeEntropyController:
    """Periodic runtime reseeding for step-time stochastic draws."""

    base_seed: int
    refresh_interval_steps: int
    supplier: RuntimeEntropySupplier | None = None
    label: str = "disabled"
    _generator: np.random.Generator = field(init=False, repr=False)
    _refresh_count: int = field(init=False, default=0)
    _last_refresh_step: int = field(init=False, default=0)
    _refresh_log: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._generator = np.random.Generator(np.random.PCG64(int(self.base_seed)))

    @property
    def generator(self) -> np.random.Generator:
        return self._generator

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def enabled(self) -> bool:
        return self.supplier is not None and self.refresh_interval_steps > 0

    def maybe_refresh(self, step_count: int) -> None:
        """Refresh the local PRNG state if the configured interval elapsed."""
        if not self.enabled:
            return

        should_refresh = self._refresh_count == 0
        if not should_refresh:
            should_refresh = (step_count - self._last_refresh_step) >= self.refresh_interval_steps
        if not should_refresh:
            return

        block = self.supplier(self._refresh_count, int(step_count))
        mixed_seed = _mix_seed(
            base_seed=self.base_seed,
            entropy_seed=block.seed64,
            refresh_index=self._refresh_count,
            step_count=step_count,
        )
        self._generator = np.random.Generator(np.random.PCG64(mixed_seed))
        self._last_refresh_step = int(step_count)
        self._refresh_log.append(
            {
                "refresh_index": int(self._refresh_count),
                "step_count": int(step_count),
                "mixed_seed64": int(mixed_seed),
                "block": block.as_metadata_record(),
            }
        )
        self._refresh_count += 1

    def random(self) -> float:
        return float(self._generator.random())

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ):
        return self._generator.normal(loc=loc, scale=scale, size=size)

    def choice(
        self,
        values: list[int] | np.ndarray,
        size: int | tuple[int, ...] | None = None,
        replace: bool = True,
    ):
        return self._generator.choice(values, size=size, replace=replace)

    def binomial(self, n: int, p: float) -> int:
        if n <= 0 or p <= 0.0:
            return 0
        if p >= 1.0:
            return int(n)
        return int(self._generator.binomial(int(n), float(p)))

    def as_metadata_record(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "label": self.label,
            "base_seed": int(self.base_seed),
            "refresh_interval_steps": int(self.refresh_interval_steps),
            "refresh_count": int(self._refresh_count),
            "refresh_log": list(self._refresh_log),
        }
