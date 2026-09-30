import unittest
from batch3_budget import fit_sample_cost, remaining_from_actual, summarize


class BudgetTests(unittest.TestCase):
    def test_startup_not_multiplied_across_source_duration(self):
        # 100 seconds startup + 0.5 seconds processing per video second.
        fit = fit_sample_cost([(60, 130), (600, 400)])
        self.assertAlmostEqual(fit['fixed_seconds'], 100)
        self.assertAlmostEqual(fit['seconds_per_source_second'], 0.5)
        total = fit['fixed_seconds'] + 36000*fit['seconds_per_source_second']
        self.assertEqual(total, 18100)
        self.assertLess(total, 130/60*36000)

    def test_no_negative_or_zero_rate_from_cold_start_noise(self):
        fit = fit_sample_cost([(60, 500), (600, 400)])
        self.assertIn('inconclusive', fit['method'])
        self.assertGreater(fit['seconds_per_source_second'], 0)

    def test_completed_m04_calibrates_only_m_and_slower_anchor_wins(self):
        done = [{'group':'Videos_M04.zip', 'source_hours':10, 'actual_elapsed_hours':2},
                {'group':'Videos_M05.zip', 'source_hours':5, 'actual_elapsed_hours':2}]
        fit = remaining_from_actual('Videos_M06.zip', 36000, done)
        self.assertEqual(fit['estimated_hours'], 4)
        self.assertEqual(fit['calibration_group'], 'Videos_M05.zip')
        self.assertIsNone(remaining_from_actual('Video_N031-N040.zip', 36000, done))

    def test_finished_group_not_counted_in_remaining_work(self):
        groups = [{'group':'Videos_M04.zip', 'status':'completed',
                   'actual_elapsed_hours':2, 'estimated_hours':100},
                  {'group':'Videos_M05.zip', 'status':'pending', 'estimated_hours':3}]
        report = summarize(groups, benchmark_hours=1, reserve_hours=4, target_hours=30)
        self.assertEqual(report['estimated_processing_hours'], 3)
        self.assertEqual(report['known_completed_hours'], 2)
        self.assertEqual(report['estimated_upper_hours'], 11.5)
        self.assertTrue(report['within_budget_estimate'])
        self.assertFalse(summarize(groups, 1, 4, 10)['within_budget_estimate'])

    def test_missing_actual_timing_cannot_claim_total_meets_deadline(self):
        report = summarize([{'group':'Videos_M04.zip', 'status':'completed',
                             'actual_elapsed_hours':None, 'estimated_hours':0}], 0, 4, 30)
        self.assertIsNone(report['within_budget_estimate'])
        self.assertEqual(report['missing_completed_timing'], ['Videos_M04.zip'])


if __name__ == '__main__':
    unittest.main()
