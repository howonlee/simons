"""Performance-oriented companion to :mod:`simon`.

``simon.py`` deliberately favors short, pedagogical implementations.  This
module preserves its public oracle and factor types, but adds the machinery
that is useful when the same structured tensors are contracted repeatedly:

* normalized, reusable tensor-train sampling environments;
* indexed variable elimination with cached min-fill/min-degree orders;
* one-branch conditional sampling and one-pass dense materialization;
* incremental, bit-packed Gaussian elimination over GF(2); and
* direct collision sampling as the default recovery method.

The fundamental complexity limits remain unchanged.  Squared TT ranks and
factor-network induced width can still be exponential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

from simon import (
    Array,
    BinaryFactor,
    BinaryTT,
    SimonFactorOracle,
    SimonOracle,
    SimonOracleTT,
    _as_real_nonnegative,
    boolean_function_factor,
    make_simon_truth_table,
)


def _normalized(vector: Array) -> Array:
    """Return a safely scaled message without changing probability ratios."""
    scale = float(np.max(np.abs(vector))) if vector.size else 0.0
    if scale and math.isfinite(scale):
        return vector / scale
    return vector


class BinaryTTSampler:
    """Reusable sampler for a nonnegative :class:`~simon.BinaryTT`.

    Suffix messages are independent of a particular draw, so they are built
    once.  Both suffix and prefix messages are normalized to avoid overflow or
    underflow on long trains; only ratios between the two branches are used.
    """

    def __init__(self, tensor: BinaryTT, *, atol: float = 1e-10) -> None:
        self.tensor = tensor
        self.atol = atol
        n = tensor.num_bits
        dtype = np.result_type(*tensor.cores) if tensor.cores else float
        self.suffix: list[Array] = [np.empty(0)] * (n + 1)
        self.suffix[n] = np.ones(1, dtype=dtype)
        for k in range(n - 1, -1, -1):
            core = tensor.cores[k]
            message = (core[:, 0, :] + core[:, 1, :]) @ self.suffix[k + 1]
            self.suffix[k] = _normalized(message)

    def sample(self, rng: random.Random | None = None) -> int:
        if rng is None:
            rng = random
        if not self.tensor.cores:
            return 0

        dtype = np.result_type(*self.tensor.cores)
        prefix = np.ones(1, dtype=dtype)
        index = 0
        for k, core in enumerate(self.tensor.cores):
            branch0 = prefix @ core[:, 0, :] @ self.suffix[k + 1]
            branch1 = prefix @ core[:, 1, :] @ self.suffix[k + 1]
            w0 = _as_real_nonnegative(
                branch0, atol=self.atol, what=f"bit {k} weight 0"
            )
            w1 = _as_real_nonnegative(
                branch1, atol=self.atol, what=f"bit {k} weight 1"
            )
            total = w0 + w1
            if total <= self.atol:
                raise ValueError(f"zero conditional mass at bit {k}")
            bit = int(rng.random() * total >= w0)
            index |= bit << k
            prefix = _normalized(prefix @ core[:, bit, :])
        return index

    def samples(self, count: int, rng: random.Random | None = None) -> list[int]:
        if count < 0:
            raise ValueError("count must be nonnegative")
        return [self.sample(rng) for _ in range(count)]


def _aligned_values(factor: BinaryFactor, variables: tuple[Hashable, ...]) -> Array:
    positions = {variable: axis for axis, variable in enumerate(factor.variables)}
    ordered = tuple(variable for variable in variables if variable in positions)
    permutation = tuple(positions[variable] for variable in ordered)
    values = factor.values
    if permutation != tuple(range(len(permutation))):
        values = np.transpose(values, permutation)
    return values.reshape(tuple(2 if variable in positions else 1 for variable in variables))


def _multiply(left: BinaryFactor, right: BinaryFactor) -> BinaryFactor:
    """Multiply factors while avoiding repeated tuple ``index`` searches."""
    left_variables = set(left.variables)
    variables = left.variables + tuple(v for v in right.variables if v not in left_variables)
    values = _aligned_values(left, variables) * _aligned_values(right, variables)
    return BinaryFactor(variables, values)


@dataclass(frozen=True)
class BinaryFactorNetwork:
    """Factor network with cached symbolic ordering and indexed elimination."""

    factors: tuple[BinaryFactor, ...]
    open_variables: tuple[Hashable, ...]
    ordering: str = "min_fill"
    _order_cache: dict[
        tuple[frozenset[Hashable], frozenset[Hashable]], tuple[Hashable, ...]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "open_variables", tuple(self.open_variables))
        if len(set(self.open_variables)) != len(self.open_variables):
            raise ValueError("open variables must be distinct")
        if self.ordering not in {"min_fill", "min_degree"}:
            raise ValueError("ordering must be 'min_fill' or 'min_degree'")

    @classmethod
    def from_network(
        cls, network, *, ordering: str = "min_fill"
    ) -> "BinaryFactorNetwork":
        return cls(tuple(network.factors), tuple(network.open_variables), ordering)

    @property
    def num_bits(self) -> int:
        return len(self.open_variables)

    def _elimination_order(
        self, fixed: frozenset[Hashable], retained: frozenset[Hashable]
    ) -> tuple[Hashable, ...]:
        key = (fixed, retained)
        cached = self._order_cache.get(key)
        if cached is not None:
            return cached

        adjacency: dict[Hashable, set[Hashable]] = {}
        serial: dict[Hashable, int] = {}
        for factor in self.factors:
            scope = [v for v in factor.variables if v not in fixed]
            for variable in scope:
                serial.setdefault(variable, len(serial))
                adjacency.setdefault(variable, set())
            for i, left in enumerate(scope):
                for right in scope[i + 1 :]:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        order: list[Hashable] = []
        candidates = set(adjacency) - retained
        while candidates:
            def score(variable: Hashable) -> tuple[int, int, int]:
                neighbors = adjacency[variable]
                if self.ordering == "min_degree":
                    fill = 0
                else:
                    neighbor_list = tuple(neighbors)
                    fill = sum(
                        right not in adjacency[left]
                        for i, left in enumerate(neighbor_list)
                        for right in neighbor_list[i + 1 :]
                    )
                return fill, len(neighbors), serial[variable]

            variable = min(candidates, key=score)
            neighbors = tuple(adjacency[variable])
            for i, left in enumerate(neighbors):
                adjacency[left].discard(variable)
                for right in neighbors[i + 1 :]:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
            del adjacency[variable]
            candidates.remove(variable)
            order.append(variable)

        result = tuple(order)
        self._order_cache[key] = result
        return result

    def _eliminate(
        self,
        evidence: Mapping[Hashable, int],
        retained: Iterable[Hashable] = (),
    ) -> BinaryFactor:
        unknown = set(evidence) - set(self.open_variables)
        if unknown:
            raise ValueError(f"evidence contains non-open variables: {unknown!r}")
        if any(bit not in (0, 1) for bit in evidence.values()):
            raise ValueError("binary evidence values must be zero or one")

        retained_set = frozenset(retained) - frozenset(evidence)
        active: dict[int, BinaryFactor] = {}
        buckets: dict[Hashable, set[int]] = {}
        next_id = 0

        def add_factor(factor: BinaryFactor) -> None:
            nonlocal next_id
            factor_id = next_id
            next_id += 1
            active[factor_id] = factor
            for variable in factor.variables:
                buckets.setdefault(variable, set()).add(factor_id)

        for factor in self.factors:
            add_factor(factor.restricted(evidence))

        order = self._elimination_order(frozenset(evidence), retained_set)
        for variable in order:
            factor_ids = tuple(buckets.get(variable, ()))
            if not factor_ids:
                continue
            bucket = [active[factor_id] for factor_id in factor_ids]
            for factor_id in factor_ids:
                factor = active.pop(factor_id)
                for scoped in factor.variables:
                    buckets[scoped].discard(factor_id)
            product = bucket[0]
            for factor in bucket[1:]:
                product = _multiply(product, factor)
            add_factor(product.sum_out(variable))

        remaining = list(active.values())
        if not remaining:
            return BinaryFactor((), np.asarray(1.0))
        product = remaining[0]
        for factor in remaining[1:]:
            product = _multiply(product, factor)
        return product

    def _contract(self, evidence: Mapping[Hashable, int]):
        product = self._eliminate(evidence)
        if product.variables:
            raise RuntimeError("internal error: variable elimination was incomplete")
        return product.values.item()

    def evaluate(self, index: int):
        if not 0 <= index < (1 << self.num_bits):
            raise IndexError(f"index {index} is outside [0, {1 << self.num_bits})")
        evidence = {v: (index >> k) & 1 for k, v in enumerate(self.open_variables)}
        return self._contract(evidence)

    def to_dense(self) -> Array:
        """Eliminate internal variables once, retaining all output axes."""
        product = self._eliminate({}, self.open_variables)
        positions = {variable: axis for axis, variable in enumerate(product.variables)}
        if set(positions) != set(self.open_variables):
            raise RuntimeError("internal error: dense result has unexpected variables")
        permutation = tuple(positions[v] for v in self.open_variables)
        values = np.transpose(product.values, permutation)
        return np.asarray(values).reshape(-1, order="F")

    def sample(
        self,
        rng: random.Random | None = None,
        *,
        atol: float = 1e-10,
    ) -> int:
        """Sample with one new contraction per bit instead of two."""
        if rng is None:
            rng = random
        evidence: dict[Hashable, int] = {}
        total = _as_real_nonnegative(
            self._contract(evidence), atol=atol, what="total weight"
        )
        index = 0
        for k, variable in enumerate(self.open_variables):
            evidence[variable] = 0
            w0 = _as_real_nonnegative(
                self._contract(evidence), atol=atol, what=f"bit {k} weight 0"
            )
            w1 = total - w0
            scale = max(1.0, total, w0)
            if w1 < -atol * scale:
                raise ValueError(f"bit {k} inferred weight 1 is negative: {w1!r}")
            w1 = max(0.0, w1)
            if total <= atol:
                raise ValueError(f"zero conditional mass at bit {k}")
            bit = int(rng.random() * total >= w0)
            evidence[variable] = bit
            total = w1 if bit else w0
            index |= bit << k
        return index

    def walsh_hadamard(self, *, normalized: bool = False) -> "BinaryFactorNetwork":
        factor = 1 / math.sqrt(2) if normalized else 1
        hadamard = factor * np.asarray([[1, 1], [1, -1]], dtype=float)
        transformed = list(self.factors)
        outputs = []
        namespace = id(self)
        for k, variable in enumerate(self.open_variables):
            output = ("complicated-walsh-output", namespace, k)
            transformed.append(BinaryFactor((variable, output), hadamard))
            outputs.append(output)
        return BinaryFactorNetwork(tuple(transformed), tuple(outputs), self.ordering)


class IncrementalGF2:
    """Incremental reduced row basis using bit-packed Python integers."""

    def __init__(self, width: int) -> None:
        if width < 0:
            raise ValueError("width must be nonnegative")
        self.width = width
        self.mask = (1 << width) - 1
        self.rows: list[int] = [0] * width
        self.rank = 0

    def add(self, row: int) -> bool:
        value = int(row) & self.mask
        while value:
            # Choosing the highest set bit ensures ``value`` has already been
            # reduced by every existing higher pivot before it is inserted.
            pivot = value.bit_length() - 1
            existing = self.rows[pivot]
            if existing:
                value ^= existing
                continue
            for column, other in enumerate(self.rows):
                if other & (1 << pivot):
                    self.rows[column] = other ^ value
            self.rows[pivot] = value
            self.rank += 1
            return True
        return False

    def nullspace(self) -> list[int]:
        free = [column for column, row in enumerate(self.rows) if not row]
        basis = []
        for column in free:
            vector = 1 << column
            # Rows use their highest bit as pivot.  Ascending pivots therefore
            # perform back-substitution after every lower variable is known.
            for pivot, row in enumerate(self.rows):
                if row and (row & vector).bit_count() & 1:
                    vector |= 1 << pivot
            basis.append(vector)
        return basis


def gf2_nullspace(rows: Iterable[int], width: int) -> list[int]:
    elimination = IncrementalGF2(width)
    for row in rows:
        elimination.add(row)
    return elimination.nullspace()


def _collision_representation(
    oracle: SimonOracle, *, ordering: str = "min_fill"
) -> BinaryTT | BinaryFactorNetwork:
    if isinstance(oracle, SimonOracleTT):
        return oracle.collision_tt()
    return BinaryFactorNetwork.from_network(oracle.collision_network(), ordering=ordering)


def _sampler(representation, *, atol: float):
    if isinstance(representation, BinaryTT):
        return BinaryTTSampler(representation, atol=atol)
    return representation


def recover_period_direct(
    oracle: SimonOracle,
    *,
    rng: random.Random | None = None,
    max_samples: int = 64,
    atol: float = 1e-10,
    ordering: str = "min_fill",
) -> int:
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    sampler = _sampler(_collision_representation(oracle, ordering=ordering), atol=atol)
    for _ in range(max_samples):
        candidate = sampler.sample(rng)
        if candidate:
            return candidate
    raise RuntimeError("only the zero shift was sampled; increase max_samples")


def simon_fourier_samples(
    oracle: SimonOracle,
    count: int,
    *,
    rng: random.Random | None = None,
    atol: float = 1e-10,
    ordering: str = "min_fill",
) -> list[int]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    spectrum = _collision_representation(oracle, ordering=ordering).walsh_hadamard()
    sampler = _sampler(spectrum, atol=atol)
    return [sampler.sample(rng) for _ in range(count)]


def recover_period_fourier(
    oracle: SimonOracle,
    *,
    rng: random.Random | None = None,
    max_samples: int | None = None,
    atol: float = 1e-10,
    ordering: str = "min_fill",
) -> int:
    n = oracle.num_bits
    if max_samples is None:
        max_samples = max(32, 8 * n)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    spectrum = _collision_representation(oracle, ordering=ordering).walsh_hadamard()
    sampler = _sampler(spectrum, atol=atol)
    equations = IncrementalGF2(n)
    for _ in range(max_samples):
        equations.add(sampler.sample(rng))
        if equations.rank == n - 1:
            nullspace = equations.nullspace()
            if len(nullspace) == 1 and nullspace[0]:
                return nullspace[0]
    raise RuntimeError(
        "samples did not determine a unique nonzero period; "
        "increase max_samples or check Simon's promise"
    )


def recover_period(oracle: SimonOracle, **kwargs) -> int:
    """Recover the period by the faster direct route under Simon's promise."""
    return recover_period_direct(oracle, **kwargs)


__all__ = [
    "Array",
    "BinaryFactor",
    "BinaryFactorNetwork",
    "BinaryTT",
    "BinaryTTSampler",
    "IncrementalGF2",
    "SimonFactorOracle",
    "SimonOracle",
    "SimonOracleTT",
    "boolean_function_factor",
    "gf2_nullspace",
    "make_simon_truth_table",
    "recover_period",
    "recover_period_direct",
    "recover_period_fourier",
    "simon_fourier_samples",
]
