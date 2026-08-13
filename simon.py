"""A classical, tensor-train implementation of Simon's algorithm.

The usual black-box problem supplies a function ``f`` with the promise

    f(x) == f(y)  iff  y == x or y == x ^ s.

This module instead takes a *white-box tensor train* for the complete relation
``F(x, y) = 1[y == f(x)]``.  Each oracle core has shape
``(left_bond, 2, output_alphabet, right_bond)`` and represents one input bit and
one local output digit.  Contracting two copies of the train constructs

    C(t) = sum(x, y) conj(F(x, y)) F(x ^ t, y).

For a Boolean relation, ``C(t)`` counts collisions.  Under Simon's promise it is
nonzero only at ``0`` and ``s``.  We can therefore recover ``s`` directly by
sampling C, or imitate Simon's quantum algorithm classically: Walsh-transform
C, sample vectors orthogonal to s, and solve a linear system over GF(2).

The running time is polynomial in the number of sites and in the *intermediate
TT bond dimensions*.  It is not a polynomial-time algorithm for arbitrary
white-box programs: their exact tensor-train ranks can be exponential.

Only NumPy is required; nothing is imported from the sibling propbits project.
Bit/site order is little-endian throughout: core k corresponds to integer bit k.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


def _as_real_nonnegative(value: complex, *, atol: float, what: str) -> float:
    """Convert a numerically real, nonnegative tensor contraction to float."""
    z = complex(value)
    scale = max(1.0, abs(z.real))
    if abs(z.imag) > atol * scale:
        raise ValueError(f"{what} is not real: {value!r}")
    if z.real < -atol * scale:
        raise ValueError(f"{what} is negative: {z.real!r}")
    return max(0.0, z.real)


@dataclass(frozen=True)
class BinaryTT:
    """A scalar tensor train with one binary physical index per site.

    ``cores[k]`` has shape ``(r_k, 2, r_{k+1})``.  Boundary bonds must be one.
    The represented value is the matrix-chain contraction selected by the bits
    of the index.
    """

    cores: tuple[Array, ...]

    def __post_init__(self) -> None:
        cores = tuple(np.asarray(core) for core in self.cores)
        if not cores:
            object.__setattr__(self, "cores", cores)
            return
        expected = 1
        for k, core in enumerate(cores):
            if core.ndim != 3 or core.shape[1] != 2:
                raise ValueError(
                    f"binary core {k} must have shape (r_left, 2, r_right)"
                )
            if core.shape[0] != expected:
                raise ValueError(
                    f"binary core {k} left bond {core.shape[0]} != {expected}"
                )
            expected = core.shape[2]
        if expected != 1:
            raise ValueError("the final TT bond must have dimension one")
        object.__setattr__(self, "cores", cores)

    @property
    def num_bits(self) -> int:
        return len(self.cores)

    @property
    def bond_dimensions(self) -> tuple[int, ...]:
        if not self.cores:
            return (1,)
        return (1,) + tuple(core.shape[2] for core in self.cores)

    def evaluate(self, index: int):
        """Evaluate one of the exponentially many entries."""
        if not 0 <= index < (1 << self.num_bits):
            raise IndexError(f"index {index} is outside [0, {1 << self.num_bits})")
        state = np.ones(1, dtype=np.result_type(*self.cores))
        for k, core in enumerate(self.cores):
            state = state @ core[:, (index >> k) & 1, :]
        return state[0]

    def total(self):
        """Sum all entries without materializing the dense tensor."""
        if not self.cores:
            return 1
        state = np.ones(1, dtype=np.result_type(*self.cores))
        for core in self.cores:
            state = state @ (core[:, 0, :] + core[:, 1, :])
        return state[0]

    def to_dense(self) -> Array:
        """Materialize the tensor; intended only for tests and small examples."""
        return np.asarray([self.evaluate(x) for x in range(1 << self.num_bits)])

    def walsh_hadamard(self, *, normalized: bool = False) -> "BinaryTT":
        """Apply a Walsh-Hadamard transform locally, preserving TT ranks."""
        factor = 1 / math.sqrt(2) if normalized else 1
        transformed = []
        for core in self.cores:
            out = np.empty_like(core, dtype=np.result_type(core, factor))
            out[:, 0, :] = factor * (core[:, 0, :] + core[:, 1, :])
            out[:, 1, :] = factor * (core[:, 0, :] - core[:, 1, :])
            transformed.append(out)
        return BinaryTT(tuple(transformed))

    def pointwise(self, other: "BinaryTT", *, conjugate_self: bool = False) -> "BinaryTT":
        """Pointwise product using Kronecker-product virtual bonds."""
        if self.num_bits != other.num_bits:
            raise ValueError("tensor trains must have the same number of bits")
        out = []
        for a, b in zip(self.cores, other.cores):
            branches = []
            for bit in (0, 1):
                left = np.conj(a[:, bit, :]) if conjugate_self else a[:, bit, :]
                branches.append(np.kron(left, b[:, bit, :]))
            out.append(np.stack(branches, axis=1))
        return BinaryTT(tuple(out))

    def magnitude_squared(self) -> "BinaryTT":
        """Return the TT whose entries are ``abs(self[index]) ** 2``."""
        return self.pointwise(self, conjugate_self=True)

    def sample(
        self,
        rng: random.Random | None = None,
        *,
        atol: float = 1e-10,
    ) -> int:
        """Sample an index proportionally to this TT's nonnegative entries.

        Suffix environments make each conditional bit probability a tensor
        contraction.  The represented entries, rather than individual core
        elements, must be real and nonnegative.
        """
        if rng is None:
            rng = random
        n = self.num_bits
        if n == 0:
            return 0

        dtype = np.result_type(*self.cores)
        suffix: list[Array] = [np.empty(0)] * (n + 1)
        suffix[n] = np.ones(1, dtype=dtype)
        for k in range(n - 1, -1, -1):
            core = self.cores[k]
            suffix[k] = (core[:, 0, :] + core[:, 1, :]) @ suffix[k + 1]

        prefix = np.ones(1, dtype=dtype)
        index = 0
        for k, core in enumerate(self.cores):
            raw0 = prefix @ core[:, 0, :] @ suffix[k + 1]
            raw1 = prefix @ core[:, 1, :] @ suffix[k + 1]
            w0 = _as_real_nonnegative(raw0, atol=atol, what=f"bit {k} weight 0")
            w1 = _as_real_nonnegative(raw1, atol=atol, what=f"bit {k} weight 1")
            total = w0 + w1
            if total <= atol:
                raise ValueError(f"zero conditional mass at bit {k}")
            bit = int(rng.random() * total >= w0)
            if bit:
                index |= 1 << k
            prefix = prefix @ core[:, bit, :]
        return index

    def sample_born(
        self, rng: random.Random | None = None, *, atol: float = 1e-10
    ) -> int:
        """Sample proportionally to squared magnitude (the Born distribution)."""
        return self.magnitude_squared().sample(rng, atol=atol)


@dataclass(frozen=True)
class SimonOracleTT:
    """TT for ``F(x, y)``, with an input bit and output digit at every site.

    Core k has shape ``(r_k, 2, d_k, r_{k+1})``.  For the common Boolean-output
    case, every ``d_k`` is two and the selected output digits are the bits of y.
    The collision construction also supports other local alphabet sizes.
    """

    cores: tuple[Array, ...]

    def __post_init__(self) -> None:
        cores = tuple(np.asarray(core) for core in self.cores)
        if not cores:
            raise ValueError("a Simon oracle must contain at least one input bit")
        expected = 1
        for k, core in enumerate(cores):
            if core.ndim != 4 or core.shape[1] != 2 or core.shape[2] < 1:
                raise ValueError(
                    f"oracle core {k} must have shape (r_left, 2, d, r_right)"
                )
            if core.shape[0] != expected:
                raise ValueError(
                    f"oracle core {k} left bond {core.shape[0]} != {expected}"
                )
            expected = core.shape[3]
        if expected != 1:
            raise ValueError("the final oracle TT bond must have dimension one")
        object.__setattr__(self, "cores", cores)

    @property
    def num_bits(self) -> int:
        return len(self.cores)

    @property
    def bond_dimensions(self) -> tuple[int, ...]:
        return (1,) + tuple(core.shape[3] for core in self.cores)

    def collision_tt(self) -> BinaryTT:
        """Construct ``C(t) = sum(x,y) conj(F(x,y)) F(x^t,y)``.

        An oracle bond of size r becomes a collision bond of size r squared.
        No input or output strings are enumerated.
        """
        result = []
        for core in self.cores:
            r_left, _, _, r_right = core.shape
            branches = []
            for shift_bit in (0, 1):
                shifted = core[:, [shift_bit, 1 ^ shift_bit], :, :]
                # Indices are (a,x,y,b) and (c,x,y,d); output is (a,c,b,d).
                transfer = np.einsum(
                    "axyb,cxyd->acbd", np.conj(core), shifted, optimize=True
                )
                branches.append(transfer.reshape(r_left * r_left, r_right * r_right))
            result.append(np.stack(branches, axis=1))
        return BinaryTT(tuple(result))

    @classmethod
    def from_truth_table(
        cls,
        outputs: Sequence[int],
        *,
        output_bits: int | None = None,
        svd_tolerance: float = 1e-12,
    ) -> "SimonOracleTT":
        """Build an exact-ish TT from a small dense truth table using TT-SVD.

        This convenience constructor materializes ``4**n`` relation entries and
        is therefore only for examples and tests.  A real white-box use should
        construct the oracle cores directly from the structured program.
        ``output_bits`` may be at most n; missing high sites use alphabet size 1.
        """
        values = [int(y) for y in outputs]
        if not values or len(values) & (len(values) - 1):
            raise ValueError("truth-table length must be a nonzero power of two")
        n = len(values).bit_length() - 1
        if any(y < 0 for y in values):
            raise ValueError("outputs must be nonnegative integers")
        needed = max(1, max(values, default=0).bit_length())
        if output_bits is None:
            output_bits = needed
        if not 0 <= output_bits <= n:
            raise ValueError("output_bits must be between zero and the input width")
        if any(y >= (1 << output_bits) for y in values):
            raise ValueError("an output does not fit in output_bits")

        alphabets = [2 if k < output_bits else 1 for k in range(n)]
        physical_dims = [2 * d for d in alphabets]
        relation = np.zeros(tuple(physical_dims), dtype=float)
        for x, y in enumerate(values):
            site_indices = []
            for k, d in enumerate(alphabets):
                x_bit = (x >> k) & 1
                y_digit = (y >> k) & 1 if d == 2 else 0
                site_indices.append(x_bit * d + y_digit)
            relation[tuple(site_indices)] = 1.0

        flat_cores = _tt_svd(relation, physical_dims, svd_tolerance)
        oracle_cores = []
        for core, d in zip(flat_cores, alphabets):
            oracle_cores.append(core.reshape(core.shape[0], 2, d, core.shape[2]))
        return cls(tuple(oracle_cores))


def _tt_svd(tensor: Array, physical_dims: Sequence[int], tolerance: float) -> list[Array]:
    """Decompose a dense tensor into open-boundary TT cores."""
    if tolerance < 0:
        raise ValueError("svd_tolerance must be nonnegative")
    work = np.asarray(tensor)
    rank_left = 1
    cores: list[Array] = []
    for physical in physical_dims[:-1]:
        matrix = work.reshape(rank_left * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        if singular.size == 0:
            rank_right = 1
        else:
            cutoff = tolerance * singular[0]
            rank_right = max(1, int(np.count_nonzero(singular > cutoff)))
        u = u[:, :rank_right]
        singular = singular[:rank_right]
        vh = vh[:rank_right, :]
        cores.append(u.reshape(rank_left, physical, rank_right))
        work = singular[:, None] * vh
        rank_left = rank_right
    cores.append(work.reshape(rank_left, physical_dims[-1], 1))
    return cores


def gf2_nullspace(rows: Iterable[int], width: int) -> list[int]:
    """Return a basis for the nullspace of bit-packed rows over GF(2)."""
    if width < 0:
        raise ValueError("width must be nonnegative")
    mask = (1 << width) - 1
    packed = [int(row) & mask for row in rows]
    matrix = np.zeros((len(packed), width), dtype=np.uint8)
    for i, row in enumerate(packed):
        for bit in range(width):
            matrix[i, bit] = (row >> bit) & 1

    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if not candidates.size:
            continue
        selected = pivot_row + int(candidates[0])
        matrix[[pivot_row, selected]] = matrix[[selected, pivot_row]]
        for row in range(matrix.shape[0]):
            if row != pivot_row and matrix[row, column]:
                matrix[row] ^= matrix[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == matrix.shape[0]:
            break

    free_columns = [c for c in range(width) if c not in set(pivot_columns)]
    basis = []
    for free in free_columns:
        vector = 1 << free
        for row, pivot in enumerate(pivot_columns):
            if matrix[row, free]:
                vector |= 1 << pivot
        basis.append(vector)
    return basis


def recover_period_direct(
    oracle: SimonOracleTT,
    *,
    rng: random.Random | None = None,
    max_samples: int = 64,
    atol: float = 1e-10,
) -> int:
    """Recover s by sampling the white-box collision tensor directly.

    Under the nonzero-period Simon promise, the distribution is uniform on
    ``{0, s}``, so each sample succeeds with probability one half.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    collision = oracle.collision_tt()
    for _ in range(max_samples):
        candidate = collision.sample(rng, atol=atol)
        if candidate:
            return candidate
    raise RuntimeError("only the zero shift was sampled; increase max_samples")


def simon_fourier_samples(
    oracle: SimonOracleTT,
    count: int,
    *,
    rng: random.Random | None = None,
    atol: float = 1e-10,
) -> list[int]:
    """Draw Simon equations z with ``z dot s == 0`` from the collision TT."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    spectrum = oracle.collision_tt().walsh_hadamard()
    return [spectrum.sample(rng, atol=atol) for _ in range(count)]


def recover_period_fourier(
    oracle: SimonOracleTT,
    *,
    rng: random.Random | None = None,
    max_samples: int | None = None,
    atol: float = 1e-10,
) -> int:
    """Classically reproduce Simon sampling and GF(2) post-processing."""
    n = oracle.num_bits
    if max_samples is None:
        max_samples = max(32, 8 * n)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    spectrum = oracle.collision_tt().walsh_hadamard()
    equations: list[int] = []
    for _ in range(max_samples):
        equations.append(spectrum.sample(rng, atol=atol))
        nullspace = gf2_nullspace(equations, n)
        if len(nullspace) == 1:
            period = nullspace[0]
            if period:
                return period
    raise RuntimeError(
        "samples did not determine a unique nonzero period; "
        "increase max_samples or check Simon's promise"
    )


def make_simon_truth_table(
    num_bits: int,
    period: int,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """Create a small random promised truth table for demonstrations/tests."""
    if num_bits < 1:
        raise ValueError("num_bits must be positive")
    if not 0 < period < (1 << num_bits):
        raise ValueError("period must be a nonzero num_bits-wide integer")
    if rng is None:
        rng = random

    representatives = [x for x in range(1 << num_bits) if x < (x ^ period)]
    labels = list(range(1 << num_bits))
    rng.shuffle(labels)
    outputs = [0] * (1 << num_bits)
    for representative, label in zip(representatives, labels):
        outputs[representative] = label
        outputs[representative ^ period] = label
    return outputs


def _demo() -> None:
    rng = random.Random(7)
    n = 5
    expected = 0b10110
    table = make_simon_truth_table(n, expected, rng=rng)
    oracle = SimonOracleTT.from_truth_table(table, output_bits=n)
    direct = recover_period_direct(oracle, rng=rng)
    fourier = recover_period_fourier(oracle, rng=rng)
    print(f"oracle bonds:    {oracle.bond_dimensions}")
    print(f"expected period: {expected:0{n}b}")
    print(f"direct recovery: {direct:0{n}b}")
    print(f"Fourier recovery:{fourier:0{n}b}")


if __name__ == "__main__":
    _demo()
