"""
Unittests for data_utils
"""


import unittest

import numpy as np
from data_utils import (
    corr_vectors,
    load_Koenig_microstate_templates,
    match_reorder_topomaps,
)


class TestDataUtils(unittest.TestCase):
    def test_load_Koenig_microstate_templates(self):
        for n in range(2, 7):
            maps, channels = load_Koenig_microstate_templates(n_states=n)
            n_chans = len(channels)
            self.assertTupleEqual(maps.shape, (n, n_chans))

    def test_corr_vectors(self):
        dims = [20, 4]
        A = np.random.rand(dims[1], dims[0])
        B = np.random.rand(dims[1], dims[0])
        for axis in [0, 1]:
            corr = corr_vectors(A, B, axis=axis)
            self.assertTupleEqual(corr.shape, (dims[axis],))

    def test_match_reorder_topomaps(self):

        FIXED_ATTRIBUTION = (3, 1, 2, 0)
        FIXED_CORR = np.array((0.14002029, 0.24210493, 0.35950477, 0.39789251))

        np.random.seed(42)
        maps1 = np.random.rand(4, 20)
        maps2 = np.random.rand(4, 20)
        att = match_reorder_topomaps(maps1, maps2, return_attribution_only=True)
        self.assertTupleEqual(att, FIXED_ATTRIBUTION)
        maps_reorder, corr = match_reorder_topomaps(
            maps1, maps2, return_correlation=True, return_attribution_only=False
        )
        np.testing.assert_equal(maps_reorder, maps1[FIXED_ATTRIBUTION, :])
        np.testing.assert_almost_equal(FIXED_CORR, corr)


if __name__ == "__main__":
    unittest.main()
