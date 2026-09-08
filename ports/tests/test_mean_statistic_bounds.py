import unittest
from mean_statistic_bounds import plan_mean_statistic, normalized_mean_bounds
from frontend import Error


class MeanStatisticBoundsTests(unittest.TestCase):
    def test_attention_producer_range(self):
        p = plan_mean_statistic(34.5625, 32, 8, 256, 4)
        self.assertLess(p["local_sum_bound"], 65504)
        self.assertGreater(p["local_sum_bound"] * 8, 65504)
        self.assertLess(p["narrowed_mean_bound"], 1300)
        self.assertTrue(p["standard_rms"])
        self.assertEqual(p["private_buffer_bytes"], 32)

    def test_mean_does_not_repair_local_overflow(self):
        with self.assertRaises(Error):
            plan_mean_statistic(33, 64, 4, 256, 4)

    def test_divisor_and_geometry_contract(self):
        for args in [
            (1, 32, 8, 255, 4),
            (1, 32, 8, True, 4),
            (1, 32.0, 8, 256, 4),
            (1, 32, 3, 256, 4),
            (1, 32, 8, 65536, 4),
            (float("nan"), 32, 8, 256, 4),
        ]:
            with self.subTest(args=args), self.assertRaises(Error):
                plan_mean_statistic(*args)

    def test_generalized_statistic_is_not_rms(self):
        self.assertFalse(plan_mean_statistic(1, 32, 8, 128, 4)["standard_rms"])

    def test_sixteen_way_range(self):
        self.assertTrue(plan_mean_statistic(34.5625, 16, 16, 256, 4)["standard_rms"])


class CorrelatedMeanBoundsTests(unittest.TestCase):
    def test_source_boundary_correlated_range(self):
        p = normalized_mean_bounds(34.5625, 1, 32, 8, 1e-6)
        self.assertGreaterEqual(p["l1_bound"], 256)
        self.assertLess(p["l1_bound"], 260)
        self.assertGreaterEqual(p["elementwise_bound"], 16)
        self.assertLess(p["elementwise_bound"], 17)
        self.assertEqual(p["absolute_mean_loss_exact"], ["3", "33554432"])

    def test_invalid_local_energy_is_not_hidden_by_normalization(self):
        with self.assertRaises(Error):
            normalized_mean_bounds(33, 1, 64, 4, 1e-6)
        with self.assertRaises(Error):
            normalized_mean_bounds(1, 1, 32, 8, 0)
