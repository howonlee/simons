"""Pedagogical tests for the classical tensor-train Simon implementation.

These examples make an important scope distinction explicit: Simon's problem
is exponentially hard for a *black-box* classical algorithm, while ``simon``
is given a white-box tensor train and can be efficient when its bond dimensions
stay small.  The final test demonstrates that favorable structured case without
ever materializing an exponentially large truth table.
"""

import random
import unittest

import numpy as np

from simon import (
    SimonOracleTT,
    gf2_nullspace,
    make_simon_truth_table,
    recover_period_direct,
    recover_period_fourier,
    simon_fourier_samples,
)


def _dot_mod_2(a: int, b: int) -> int:
    """GF(2) inner product for bit-packed vectors."""
    return (a & b).bit_count() & 1


def _rank_one_bit_deletion_oracle(num_bits: int, hidden_bit: int) -> SimonOracleTT:
    """Represent f(x) = x with one bit deleted as a rank-one oracle TT.

    Deleting bit k makes exactly x and x xor 2**k share an output, so the
    Simon period is 2**k.  Every TT bond has dimension one.
    """
    cores = []
    for bit in range(num_bits):
        if bit == hidden_bit:
            # This input bit is ignored and this site has no output bit.
            core = np.ones((1, 2, 1, 1), dtype=float)
        else:
            # Local relation y_bit == x_bit.
            core = np.zeros((1, 2, 2, 1), dtype=float)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
        cores.append(core)
    return SimonOracleTT(tuple(cores))


class SimonPedagogicalTests(unittest.TestCase):
    def test_truth_table_has_exactly_the_promised_pairs(self) -> None:
        """A small concrete example of Simon's two-to-one promise."""
        num_bits = 4
        period = 0b1011
        table = make_simon_truth_table(num_bits, period, rng=random.Random(1))

        for x in range(1 << num_bits):
            for y in range(1 << num_bits):
                self.assertEqual(
                    table[x] == table[y],
                    y in (x, x ^ period),
                )

    def test_collision_tensor_exposes_only_zero_and_the_period(self) -> None:
        """The contraction turns period finding into two nonzero shifts."""
        num_bits = 4
        period = 0b1011
        table = make_simon_truth_table(num_bits, period, rng=random.Random(2))
        oracle = SimonOracleTT.from_truth_table(table, output_bits=num_bits)

        collisions = oracle.collision_tt().to_dense()
        expected = np.zeros(1 << num_bits)
        expected[0] = expected[period] = 1 << num_bits

        np.testing.assert_allclose(collisions, expected, atol=1e-9)

    def test_walsh_spectrum_gives_simons_linear_equations(self) -> None:
        """Fourier samples z all satisfy z dot s = 0, as in Simon's algorithm."""
        num_bits = 4
        period = 0b1011
        table = make_simon_truth_table(num_bits, period, rng=random.Random(3))
        oracle = SimonOracleTT.from_truth_table(table, output_bits=num_bits)

        spectrum = oracle.collision_tt().walsh_hadamard().to_dense()
        for z, weight in enumerate(spectrum):
            expected = 2 * (1 << num_bits) if _dot_mod_2(z, period) == 0 else 0
            np.testing.assert_allclose(weight, expected, atol=1e-8)

        samples = simon_fourier_samples(oracle, 32, rng=random.Random(4))
        self.assertTrue(all(_dot_mod_2(z, period) == 0 for z in samples))
        self.assertEqual(gf2_nullspace(samples, num_bits), [period])

    def test_both_classical_recovery_routes_find_the_period(self) -> None:
        """Direct collision sampling and Simon-style post-processing agree."""
        num_bits = 5
        period = 0b10110
        table = make_simon_truth_table(num_bits, period, rng=random.Random(5))
        oracle = SimonOracleTT.from_truth_table(table, output_bits=num_bits)

        self.assertEqual(
            recover_period_direct(oracle, rng=random.Random(6)), period
        )
        self.assertEqual(
            recover_period_fourier(oracle, rng=random.Random(7)), period
        )

    def test_compact_white_box_oracle_is_viable_at_larger_width(self) -> None:
        """A 40-bit structured instance remains tiny instead of storing 2**40 rows."""
        num_bits = 40
        hidden_bit = 31
        period = 1 << hidden_bit
        oracle = _rank_one_bit_deletion_oracle(num_bits, hidden_bit)

        self.assertEqual(oracle.bond_dimensions, (1,) * (num_bits + 1))
        self.assertEqual(
            oracle.collision_tt().bond_dimensions, (1,) * (num_bits + 1)
        )
        self.assertEqual(
            recover_period_direct(oracle, rng=random.Random(8)), period
        )
        self.assertEqual(
            recover_period_fourier(oracle, rng=random.Random(9)), period
        )


if __name__ == "__main__":
    unittest.main()
