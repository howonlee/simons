"""Chaskey/Even--Mansour experiments for the structured Simon implementation.

This module builds local Boolean factors for the parameterized Chaskey
permutation and for the one-key Even--Mansour Simon function

    f_s(x) = P(x) xor P(x xor s) xor s.

It is primarily a structural experiment.  The builder knows ``s`` in order to
construct the complete white-box relation; ordinary black-box access to an
Even--Mansour encryption oracle does not provide such a factorization.

The command-line profiler symbolically follows the same min-fill variable
elimination order.  It stops at a configurable scope size before allocating an
exponential dense factor.  Exact recovery is available for cases whose profile
is small enough::

    python3 chaskey_em.py --word-bits 1 --rounds 1 --recover
    python3 chaskey_em.py --ladder --rounds 4

This is kind of lame, frankly, but what can be done
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
from typing import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from complicated_simon import (
    BinaryFactor,
    BinaryFactorNetwork,
    SimonFactorOracle,
    boolean_function_factor,
)


def _full_adder_table() -> np.ndarray:
    table = np.zeros((2, 2, 2, 2, 2), dtype=float)
    for left, right, carry_in in np.ndindex(2, 2, 2):
        total = left + right + carry_in
        table[left, right, carry_in, total & 1, total >> 1] = 1
    return table


def _last_adder_bit_table() -> np.ndarray:
    table = np.zeros((2, 2, 2, 2), dtype=float)
    for left, right, carry_in in np.ndindex(2, 2, 2):
        table[left, right, carry_in, left ^ right ^ carry_in] = 1
    return table


_FULL_ADDER = _full_adder_table()
_LAST_ADDER_BIT = _last_adder_bit_table()


class _Circuit:
    """Small helper that emits relation factors for a Boolean circuit."""

    def __init__(self) -> None:
        self.factors: list[BinaryFactor] = []
        self._serial = 0

    def fresh(self, label: Hashable) -> Hashable:
        variable = ("chaskey-wire", self._serial, label)
        self._serial += 1
        return variable

    def constant(self, variable: Hashable, value: int) -> None:
        table = np.zeros(2, dtype=float)
        table[int(value)] = 1
        self.factors.append(BinaryFactor((variable,), table))

    def xor_bit(self, left: Hashable, right: Hashable, label: Hashable) -> Hashable:
        output = self.fresh(label)
        self.factors.append(
            boolean_function_factor((left, right), output, lambda a, b: a ^ b)
        )
        return output

    def xor_constant(self, value: Hashable, bit: int, label: Hashable) -> Hashable:
        if not bit:
            return value
        output = self.fresh(label)
        self.factors.append(
            boolean_function_factor((value,), output, lambda a: a ^ 1)
        )
        return output

    def xor_words(
        self,
        left: Sequence[Hashable],
        right: Sequence[Hashable],
        label: Hashable,
    ) -> tuple[Hashable, ...]:
        if len(left) != len(right):
            raise ValueError("word widths differ")
        return tuple(
            self.xor_bit(a, b, (label, bit))
            for bit, (a, b) in enumerate(zip(left, right))
        )

    def add_words(
        self,
        left: Sequence[Hashable],
        right: Sequence[Hashable],
        label: Hashable,
    ) -> tuple[Hashable, ...]:
        """Emit a little-endian ripple-carry addition modulo the word size."""
        if len(left) != len(right) or not left:
            raise ValueError("addition needs two nonempty, equally wide words")
        carry = self.fresh((label, "carry", 0))
        self.constant(carry, 0)
        output = []
        for bit, (a, b) in enumerate(zip(left, right)):
            result = self.fresh((label, "sum", bit))
            if bit + 1 == len(left):
                self.factors.append(
                    BinaryFactor((a, b, carry, result), _LAST_ADDER_BIT)
                )
            else:
                carry_out = self.fresh((label, "carry", bit + 1))
                self.factors.append(
                    BinaryFactor((a, b, carry, result, carry_out), _FULL_ADDER)
                )
                carry = carry_out
            output.append(result)
        return tuple(output)


def _rotl_word(word: Sequence[Hashable], amount: int) -> tuple[Hashable, ...]:
    """Rewire a little-endian word through a left rotation."""
    width = len(word)
    amount %= width
    return tuple(word[(bit - amount) % width] for bit in range(width))


def _chaskey_circuit(
    circuit: _Circuit,
    words: Sequence[Sequence[Hashable]],
    rounds: int,
    label: Hashable,
) -> tuple[tuple[Hashable, ...], ...]:
    if len(words) != 4 or len({len(word) for word in words}) != 1:
        raise ValueError("Chaskey needs four equally wide words")
    if rounds < 0:
        raise ValueError("round count must be nonnegative")
    v0, v1, v2, v3 = (tuple(word) for word in words)
    for round_index in range(rounds):
        step = (label, "round", round_index)
        v0 = circuit.add_words(v0, v1, (step, "add-01"))
        v1 = circuit.xor_words(_rotl_word(v1, 5), v0, (step, "xor-10"))
        v0 = _rotl_word(v0, 16)

        v2 = circuit.add_words(v2, v3, (step, "add-23"))
        v3 = circuit.xor_words(_rotl_word(v3, 8), v2, (step, "xor-32"))

        v0 = circuit.add_words(v0, v3, (step, "add-03"))
        v3 = circuit.xor_words(_rotl_word(v3, 13), v0, (step, "xor-30"))

        v2 = circuit.add_words(v2, v1, (step, "add-21"))
        v1 = circuit.xor_words(_rotl_word(v1, 7), v2, (step, "xor-12"))
        v2 = _rotl_word(v2, 16)
    return v0, v1, v2, v3


def make_chaskey_em_oracle(
    word_bits: int,
    rounds: int,
    period: int,
) -> SimonFactorOracle:
    """Build factors for ``P(x) xor P(x xor period) xor period``.

    Four ``word_bits``-wide words form the state.  Rotation amounts are the
    standard Chaskey constants reduced modulo ``word_bits`` for scaled models.
    """
    if word_bits < 1:
        raise ValueError("word_bits must be positive")
    state_bits = 4 * word_bits
    if not 0 < period < (1 << state_bits):
        raise ValueError(f"period must be a nonzero {state_bits}-bit integer")

    circuit = _Circuit()
    inputs = tuple(("chaskey-input", bit) for bit in range(state_bits))
    input_words = tuple(
        inputs[word * word_bits : (word + 1) * word_bits] for word in range(4)
    )
    shifted = tuple(
        circuit.xor_constant(variable, (period >> bit) & 1, ("secret-shift", bit))
        for bit, variable in enumerate(inputs)
    )
    shifted_words = tuple(
        shifted[word * word_bits : (word + 1) * word_bits] for word in range(4)
    )

    plain_output = _chaskey_circuit(circuit, input_words, rounds, "plain")
    shifted_output = _chaskey_circuit(circuit, shifted_words, rounds, "shifted")
    plain_bits = tuple(bit for word in plain_output for bit in word)
    shifted_bits = tuple(bit for word in shifted_output for bit in word)

    outputs = tuple(("chaskey-simon-output", bit) for bit in range(state_bits))
    for bit, (left, right, output) in enumerate(
        zip(plain_bits, shifted_bits, outputs)
    ):
        secret_bit = (period >> bit) & 1
        circuit.factors.append(
            boolean_function_factor(
                (left, right),
                output,
                lambda a, b, constant=secret_bit: a ^ b ^ constant,
            )
        )
    return SimonFactorOracle(tuple(circuit.factors), inputs, outputs)


def chaskey_permute(value: int, word_bits: int, rounds: int) -> int:
    """Integer reference implementation of the parameterized permutation."""
    if word_bits < 1 or rounds < 0:
        raise ValueError("word_bits must be positive and rounds nonnegative")
    mask = (1 << word_bits) - 1
    words = [(value >> (word_bits * k)) & mask for k in range(4)]

    def rotl(word: int, amount: int) -> int:
        amount %= word_bits
        if amount == 0:
            return word
        return ((word << amount) | (word >> (word_bits - amount))) & mask

    v0, v1, v2, v3 = words
    for _ in range(rounds):
        v0 = (v0 + v1) & mask
        v1 = rotl(v1, 5) ^ v0
        v0 = rotl(v0, 16)
        v2 = (v2 + v3) & mask
        v3 = rotl(v3, 8) ^ v2
        v0 = (v0 + v3) & mask
        v3 = rotl(v3, 13) ^ v0
        v2 = (v2 + v1) & mask
        v1 = rotl(v1, 7) ^ v2
        v2 = rotl(v2, 16)
    return (
        v0
        | (v1 << word_bits)
        | (v2 << (2 * word_bits))
        | (v3 << (3 * word_bits))
    )


def chaskey_em_simon_function(
    x: int, word_bits: int, rounds: int, period: int
) -> int:
    """Evaluate the Simon function represented by :func:`make_chaskey_em_oracle`."""
    return (
        chaskey_permute(x, word_bits, rounds)
        ^ chaskey_permute(x ^ period, word_bits, rounds)
        ^ period
    )


@dataclass(frozen=True)
class EliminationProfile:
    variables: int
    factors: int
    eliminated: int
    maximum_scope: int
    completed: bool

    @property
    def maximum_entries(self) -> int:
        return 1 << self.maximum_scope


def elimination_profile(
    network: BinaryFactorNetwork,
    evidence: Mapping[Hashable, int] | None = None,
    *,
    stop_scope: int = 30,
) -> EliminationProfile:
    """Symbolically profile min-fill elimination without dense contractions.

    ``maximum_scope`` includes the variable being eliminated, matching the
    largest bucket product that dense elimination would form.  Profiling stops
    as soon as this exceeds ``stop_scope``.
    """
    if stop_scope < 1:
        raise ValueError("stop_scope must be positive")
    fixed = set(evidence or {})
    adjacency: dict[Hashable, set[Hashable]] = {}
    order: dict[Hashable, int] = {}
    maximum_scope = 0

    for factor in network.factors:
        scope = [variable for variable in factor.variables if variable not in fixed]
        maximum_scope = max(maximum_scope, len(scope))
        for variable in scope:
            if variable not in order:
                order[variable] = len(order)
            adjacency.setdefault(variable, set())
        for index, left in enumerate(scope):
            for right in scope[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    initial_variables = len(adjacency)
    eliminated = 0
    while adjacency:
        def score(variable: Hashable) -> tuple[int, int, int]:
            neighbors = tuple(adjacency[variable])
            fill = sum(
                right not in adjacency[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1 :]
            )
            return fill, len(neighbors), order[variable]

        variable = min(adjacency, key=score)
        neighbors = tuple(adjacency[variable])
        scope_size = len(neighbors) + 1
        maximum_scope = max(maximum_scope, scope_size)
        if scope_size > stop_scope:
            return EliminationProfile(
                initial_variables,
                len(network.factors),
                eliminated,
                maximum_scope,
                False,
            )

        for index, left in enumerate(neighbors):
            adjacency[left].discard(variable)
            for right in neighbors[index + 1 :]:
                if right != left:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        del adjacency[variable]
        eliminated += 1

    return EliminationProfile(
        initial_variables,
        len(network.factors),
        eliminated,
        maximum_scope,
        True,
    )


def profile_instance(
    word_bits: int,
    rounds: int,
    period: int,
    *,
    stop_scope: int,
) -> tuple[SimonFactorOracle, BinaryFactorNetwork, EliminationProfile]:
    oracle = make_chaskey_em_oracle(word_bits, rounds, period)
    collision = BinaryFactorNetwork.from_network(oracle.collision_network())
    # The first sampler contraction fixes the low shift bit and sums all other
    # open variables.  That is more representative than profiling an entry.
    evidence = {collision.open_variables[0]: 0}
    profile = elimination_profile(collision, evidence, stop_scope=stop_scope)
    return oracle, collision, profile


def recover_verified_period(
    oracle: SimonFactorOracle,
    evaluator: Callable[[int], int],
    *,
    rng: random.Random,
    verification_samples: int = 32,
    max_samples: int = 64,
) -> int:
    """Sample collision shifts and retain only a verified global period.

    Small EM-derived Simon functions have accidental collisions, so returning
    the first nonzero collision (the ideal Simon-promise shortcut) is not
    reliable.  A candidate is cheap to verify with ordinary function queries.
    For at most 16 input bits this routine verifies exhaustively.
    """
    collision = BinaryFactorNetwork.from_network(oracle.collision_network())
    width = oracle.num_bits
    exhaustive = width <= 16
    for _ in range(max_samples):
        candidate = collision.sample(rng)
        if not candidate:
            continue
        probes = range(1 << width) if exhaustive else (
            rng.randrange(1 << width) for _ in range(verification_samples)
        )
        if all(evaluator(x) == evaluator(x ^ candidate) for x in probes):
            return candidate
    raise RuntimeError("no sampled collision passed global-period verification")


def _default_period(state_bits: int) -> int:
    # A deterministic, nonlocal period that touches all four words.
    return ((1 << state_bits) - 1) // 3


def _print_profile(word_bits: int, rounds: int, profile: EliminationProfile) -> None:
    status = (
        "complete"
        if profile.completed
        else f"> limit after {profile.eliminated} vars"
    )
    entries = (
        str(profile.maximum_entries)
        if profile.maximum_scope < 63
        else f"2^{profile.maximum_scope}"
    )
    print(
        f"w={word_bits:2d}  n={4 * word_bits:3d}  rounds={rounds:2d}  "
        f"factors={profile.factors:5d}  max_scope={profile.maximum_scope:2d}  "
        f"entries={entries:>10}  {status}"
    )


def run_ladder(
    word_sizes: Iterable[int],
    rounds: Iterable[int],
    *,
    stop_scope: int,
) -> None:
    for word_bits in word_sizes:
        period = _default_period(4 * word_bits)
        for round_count in rounds:
            _, _, profile = profile_instance(
                word_bits, round_count, period, stop_scope=stop_scope
            )
            _print_profile(word_bits, round_count, profile)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-bits", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--period", type=lambda value: int(value, 0))
    parser.add_argument("--profile-limit", type=int, default=30)
    parser.add_argument(
        "--ladder",
        action="store_true",
        help="profile word widths 4, 8, 16, 32 and rounds 1 through --rounds",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="run exact sampled-and-verified recovery when profile scope is <= 24",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.ladder:
        run_ladder(
            (4, 8, 16, 32),
            range(1, args.rounds + 1),
            stop_scope=args.profile_limit,
        )
        return 0

    state_bits = 4 * args.word_bits
    period = args.period
    if period is None:
        period = _default_period(state_bits)
    oracle, _, profile = profile_instance(
        args.word_bits,
        args.rounds,
        period,
        stop_scope=args.profile_limit,
    )
    _print_profile(args.word_bits, args.rounds, profile)
    print(f"period=0x{period:0{(state_bits + 3) // 4}x}")

    if args.recover:
        if not profile.completed or profile.maximum_scope > 24:
            parser.error(
                "exact recovery refused: profile is incomplete or exceeds scope 24"
            )
        recovered = recover_verified_period(
            oracle,
            lambda x: chaskey_em_simon_function(
                x, args.word_bits, args.rounds, period
            ),
            rng=random.Random(args.seed),
        )
        print(f"recovered=0x{recovered:0{(state_bits + 3) // 4}x}")
        print(f"matches={recovered == period}")
        if recovered != period:
            print("note=the scaled function has multiple global periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
