"""Classical structured-tensor implementations of Simon's algorithm.

The usual black-box problem supplies a function ``f`` with the promise

    f(x) == f(y)  iff  y == x or y == x ^ s.

The first representation takes a *white-box tensor train* for the complete
relation ``F(x, y) = 1[y == f(x)]``.  Each oracle core has shape
``(left_bond, 2, output_alphabet, right_bond)`` and represents one input bit and
one local output digit.  Contracting two copies of the train constructs

    C(t) = sum(x, y) conj(F(x, y)) F(x ^ t, y).

For a Boolean relation, ``C(t)`` counts collisions.  Under Simon's promise it is
nonzero only at ``0`` and ``s``.  We can therefore recover ``s`` directly by
sampling C, or imitate Simon's quantum algorithm classically: Walsh-transform
C, sample vectors orthogonal to s, and solve a linear system over GF(2).

The second representation takes local factors for a Boolean circuit or
constraint network.  It forms the doubled collision network directly and uses
variable elimination, without first compressing the whole oracle relation into
a one-dimensional train.  This replaces the TT's pathwidth restriction by an
induced-width/treewidth restriction and can handle very different structures.

Neither route is a polynomial-time algorithm for arbitrary white-box programs:
exact TT ranks and variable-elimination intermediates can both be exponential.

Bit/site order is little-endian throughout: core k corresponds to integer bit k.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Hashable, Iterable, Mapping, Sequence

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
class BinaryFactor:
    """A dense local factor over named binary variables.

    Unlike a TT core, a factor may connect any small set of variables.  A
    collection of these factors is useful for describing a Boolean circuit or
    constraint system without first forcing it through a one-dimensional cut.
    """

    variables: tuple[Hashable, ...]
    values: Array

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if len(set(variables)) != len(variables):
            raise ValueError("a factor cannot contain a variable twice")
        values = np.asarray(self.values)
        if values.shape != (2,) * len(variables):
            raise ValueError(
                f"factor shape {values.shape} does not match "
                f"{len(variables)} binary variables"
            )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "values", values)

    def renamed(self, names: Mapping[Hashable, Hashable]) -> "BinaryFactor":
        """Return this factor with variables renamed according to ``names``."""
        return BinaryFactor(tuple(names.get(v, v) for v in self.variables), self.values)

    def restricted(self, evidence: Mapping[Hashable, int]) -> "BinaryFactor":
        """Fix variables present in ``evidence`` and remove their axes."""
        index = tuple(evidence.get(v, slice(None)) for v in self.variables)
        remaining = tuple(v for v in self.variables if v not in evidence)
        return BinaryFactor(remaining, self.values[index])

    def multiply(self, other: "BinaryFactor") -> "BinaryFactor":
        """Multiply factors, aligning axes that name the same variable."""
        variables = self.variables + tuple(
            v for v in other.variables if v not in self.variables
        )

        def aligned(factor: "BinaryFactor") -> Array:
            ordered = tuple(v for v in variables if v in factor.variables)
            permutation = tuple(factor.variables.index(v) for v in ordered)
            values = factor.values
            if permutation != tuple(range(len(permutation))):
                values = np.transpose(values, permutation)
            shape = tuple(2 if v in factor.variables else 1 for v in variables)
            return values.reshape(shape)

        return BinaryFactor(variables, aligned(self) * aligned(other))

    def sum_out(self, variable: Hashable) -> "BinaryFactor":
        """Sum one variable out of this factor."""
        if variable not in self.variables:
            return self
        axis = self.variables.index(variable)
        variables = self.variables[:axis] + self.variables[axis + 1 :]
        return BinaryFactor(variables, self.values.sum(axis=axis))


def boolean_function_factor(
    inputs: Sequence[Hashable],
    output: Hashable,
    function: Callable[..., int],
    *,
    dtype=float,
) -> BinaryFactor:
    """Create the relation factor ``output == function(*inputs)``.

    This convenience constructor makes AND, XOR, NOT, S-box-bit, and other
    small Boolean circuit constraints easy to express.  Large truth tables are
    deliberately not hidden: the factor has ``2**(len(inputs) + 1)`` entries.
    """
    inputs = tuple(inputs)
    if output in inputs or len(set(inputs)) != len(inputs):
        raise ValueError("gate input and output variables must be distinct")
    table = np.zeros((2,) * (len(inputs) + 1), dtype=dtype)
    for index in np.ndindex((2,) * len(inputs)):
        value = int(function(*index))
        if value not in (0, 1):
            raise ValueError("a Boolean gate must return zero or one")
        table[index + (value,)] = 1
    return BinaryFactor(inputs + (output,), table)


@dataclass(frozen=True)
class BinaryFactorNetwork:
    """An open binary tensor network contracted by exact variable elimination.

    Complexity is exponential in induced width (treewidth for a good order),
    rather than in the pathwidth imposed by a tensor train.  This makes the
    representation complementary to :class:`BinaryTT`: either may be much
    smaller than the other on a particular structured problem.
    """

    factors: tuple[BinaryFactor, ...]
    open_variables: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        open_variables = tuple(self.open_variables)
        if len(set(open_variables)) != len(open_variables):
            raise ValueError("open variables must be distinct")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "open_variables", open_variables)

    @property
    def num_bits(self) -> int:
        return len(self.open_variables)

    def _contract(self, evidence: Mapping[Hashable, int]):
        unknown = set(evidence) - set(self.open_variables)
        if unknown:
            raise ValueError(f"evidence contains non-open variables: {unknown!r}")
        if any(bit not in (0, 1) for bit in evidence.values()):
            raise ValueError("binary evidence values must be zero or one")

        factors = [factor.restricted(evidence) for factor in self.factors]
        variables: list[Hashable] = []
        for factor in factors:
            for variable in factor.variables:
                if variable not in variables:
                    variables.append(variable)

        # Greedy min-scope elimination.  It is inexpensive and avoids imposing
        # the bit order as a path decomposition, though it is not always the
        # globally optimal treewidth order.
        while variables:
            def elimination_width(variable: Hashable) -> int:
                scope: set[Hashable] = set()
                for factor in factors:
                    if variable in factor.variables:
                        scope.update(factor.variables)
                return len(scope)

            variable = min(variables, key=elimination_width)
            variables.remove(variable)
            bucket = [factor for factor in factors if variable in factor.variables]
            if not bucket:
                continue
            factors = [factor for factor in factors if variable not in factor.variables]
            product = bucket[0]
            for factor in bucket[1:]:
                product = product.multiply(factor)
            factors.append(product.sum_out(variable))

        if not factors:
            return 1.0
        product = factors[0]
        for factor in factors[1:]:
            product = product.multiply(factor)
        if product.variables:
            raise RuntimeError("internal error: variable elimination was incomplete")
        return product.values.item()

    def evaluate(self, index: int):
        """Evaluate one entry without materializing the open tensor."""
        if not 0 <= index < (1 << self.num_bits):
            raise IndexError(f"index {index} is outside [0, {1 << self.num_bits})")
        evidence = {v: (index >> k) & 1 for k, v in enumerate(self.open_variables)}
        return self._contract(evidence)

    def to_dense(self) -> Array:
        """Materialize the open tensor; intended for tests and small networks."""
        return np.asarray([self.evaluate(x) for x in range(1 << self.num_bits)])

    def sample(
        self,
        rng: random.Random | None = None,
        *,
        atol: float = 1e-10,
    ) -> int:
        """Sample nonnegative entries using conditional network contractions."""
        if rng is None:
            rng = random
        evidence: dict[Hashable, int] = {}
        index = 0
        for k, variable in enumerate(self.open_variables):
            evidence[variable] = 0
            w0 = _as_real_nonnegative(
                self._contract(evidence), atol=atol, what=f"bit {k} weight 0"
            )
            evidence[variable] = 1
            w1 = _as_real_nonnegative(
                self._contract(evidence), atol=atol, what=f"bit {k} weight 1"
            )
            total = w0 + w1
            if total <= atol:
                raise ValueError(f"zero conditional mass at bit {k}")
            bit = int(rng.random() * total >= w0)
            evidence[variable] = bit
            index |= bit << k
        return index

    def walsh_hadamard(self, *, normalized: bool = False) -> "BinaryFactorNetwork":
        """Attach local Walsh factors and expose their other legs as outputs."""
        factor = 1 / math.sqrt(2) if normalized else 1
        hadamard = factor * np.asarray([[1, 1], [1, -1]], dtype=float)
        transformed = list(self.factors)
        outputs = []
        for k, variable in enumerate(self.open_variables):
            output = ("walsh-output", id(self), k)
            transformed.append(BinaryFactor((variable, output), hadamard))
            outputs.append(output)
        return BinaryFactorNetwork(tuple(transformed), tuple(outputs))


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


@dataclass(frozen=True)
class SimonFactorOracle:
    """A factored relation for Simon's oracle, optionally with internal wires.

    The product of ``factors``, summed over variables that are neither inputs
    nor outputs, represents ``F(x, y)``.  For ordinary Boolean circuits every
    internal wire has one satisfying value, so local gate-relation factors give
    exactly ``1[y == f(x)]``.

    Collision construction duplicates the circuit, shares its output, and adds
    local constraints ``x_prime == x xor shift``.  It therefore avoids ever
    constructing a TT for the complete input/output truth table.
    """

    factors: tuple[BinaryFactor, ...]
    input_variables: tuple[Hashable, ...]
    output_variables: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        inputs = tuple(self.input_variables)
        outputs = tuple(self.output_variables)
        if not inputs:
            raise ValueError("a Simon oracle must contain at least one input bit")
        if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
            raise ValueError("input and output variables must each be distinct")
        if set(inputs) & set(outputs):
            raise ValueError("input and output variables must be disjoint")
        present = {v for factor in factors for v in factor.variables}
        missing_outputs = set(outputs) - present
        if missing_outputs:
            raise ValueError(
                f"output variables do not occur in a factor: {missing_outputs!r}"
            )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "input_variables", inputs)
        object.__setattr__(self, "output_variables", outputs)

    @property
    def num_bits(self) -> int:
        return len(self.input_variables)

    def collision_network(self) -> BinaryFactorNetwork:
        """Build an open factor network for ``sum(x,y) conj(F(x,y)) F(x^t,y)``."""
        inputs = set(self.input_variables)
        outputs = set(self.output_variables)
        all_variables = {v for factor in self.factors for v in factor.variables}

        x = {v: ("collision-input", k) for k, v in enumerate(self.input_variables)}
        xp = {
            v: ("collision-shifted-input", k)
            for k, v in enumerate(self.input_variables)
        }
        y = {v: ("collision-output", k) for k, v in enumerate(self.output_variables)}
        first = {}
        second = {}
        for variable in all_variables:
            if variable in inputs:
                first[variable] = x[variable]
                second[variable] = xp[variable]
            elif variable in outputs:
                first[variable] = second[variable] = y[variable]
            else:
                first[variable] = ("collision-first-internal", variable)
                second[variable] = ("collision-second-internal", variable)

        factors = [
            BinaryFactor(factor.renamed(first).variables, np.conj(factor.values))
            for factor in self.factors
        ]
        factors.extend(factor.renamed(second) for factor in self.factors)

        shifts = []
        xor_relation = np.zeros((2, 2, 2), dtype=float)
        for left, right, shift in np.ndindex(2, 2, 2):
            xor_relation[left, right, shift] = float(right == (left ^ shift))
        for k, variable in enumerate(self.input_variables):
            shift = ("collision-open-shift", k)
            shifts.append(shift)
            factors.append(BinaryFactor((x[variable], xp[variable], shift), xor_relation))
        return BinaryFactorNetwork(tuple(factors), tuple(shifts))


SimonOracle = SimonOracleTT | SimonFactorOracle


def _collision_representation(
    oracle: SimonOracle,
) -> BinaryTT | BinaryFactorNetwork:
    if isinstance(oracle, SimonOracleTT):
        return oracle.collision_tt()
    return oracle.collision_network()


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
    oracle: SimonOracle,
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
    collision = _collision_representation(oracle)
    for _ in range(max_samples):
        candidate = collision.sample(rng, atol=atol)
        if candidate:
            return candidate
    raise RuntimeError("only the zero shift was sampled; increase max_samples")


def simon_fourier_samples(
    oracle: SimonOracle,
    count: int,
    *,
    rng: random.Random | None = None,
    atol: float = 1e-10,
) -> list[int]:
    """Draw Simon equations z with ``z dot s == 0`` from the collision TT."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    spectrum = _collision_representation(oracle).walsh_hadamard()
    return [spectrum.sample(rng, atol=atol) for _ in range(count)]


def recover_period_fourier(
    oracle: SimonOracle,
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

    spectrum = _collision_representation(oracle).walsh_hadamard()
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
