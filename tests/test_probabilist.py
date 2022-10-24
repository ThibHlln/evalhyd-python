import unittest
import numpy

import evalhyd


# load some predicted and observed streamflow
_prd = (
    numpy.genfromtxt("./data/q_prd.csv", delimiter=',')
    [:5, :][numpy.newaxis, numpy.newaxis, ...]
)
_obs = numpy.genfromtxt("./data/q_obs.csv", delimiter=',')[numpy.newaxis, :]


class TestMetrics(unittest.TestCase):

    expected_thr = {
        'BS':
            [[[[[0.1081672, 0.073954980, 0.08681672, numpy.nan]]]]],
        'BSS':
            [[[[[0.56240422, 0.66612211, 0.56288391, numpy.nan]]]]],
        'BS_CRD':
            [[[[[[0.01335634, 0.15237434, 0.24718520],
                 [0.00550861, 0.15305671, 0.22150309],
                 [0.00753750, 0.11933328, 0.19861250],
                 [numpy.nan, numpy.nan, numpy.nan]]]]]],
        'BS_LBD':
            [[[[[[0.01244569, 0.14933386, 0.24505537],
                 [0.00801337, 0.14745568, 0.21339730],
                 [0.01719462, 0.10479711, 0.17441921],
                 [numpy.nan, numpy.nan, numpy.nan]]]]]]
    }

    expected_qtl = {
        'QS':
            [[[[[321.1607717,  294.3494105,  265.70418006,
                 236.15648446, 206.03965702]]]]],
        'CRPS':
            [[[[176.63504823]]]]
    }

    def test_threshold_metrics(self):
        thr = numpy.array([[690, 534, 445, numpy.nan]])
        for metric in self.expected_thr.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric], thr)[0],
                    self.expected_thr[metric]
                )

    def test_quantile_metrics(self):
        for metric in self.expected_qtl.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric])[0],
                    self.expected_qtl[metric]
                )


class TestDecomposition(unittest.TestCase):

    def test_brier_calibration_refinement(self):
        thr = numpy.array([[690, 534, 445]])
        bs, = evalhyd.evalp(_obs, _prd, ["BS"], thr)
        bs_crd, = evalhyd.evalp(_obs, _prd, ["BS_CRD"], thr)
        numpy.testing.assert_almost_equal(
            bs, bs_crd[..., 0] - bs_crd[..., 1] + bs_crd[..., 2]
        )

    def test_brier_likelihood_base_rate(self):
        thr = numpy.array([[690, 534, 445]])
        bs, = evalhyd.evalp(_obs, _prd, ["BS"], thr)
        bs_lbd, = evalhyd.evalp(_obs, _prd, ["BS_LBD"], thr)
        numpy.testing.assert_almost_equal(
            bs, bs_lbd[..., 0] - bs_lbd[..., 1] + bs_lbd[..., 2]
        )


class TestMasking(unittest.TestCase):

    def test_masks(self):
        msk = numpy.ones((_prd.shape[0], _prd.shape[1], 1, _prd.shape[3]), dtype=bool)
        msk[..., :99] = False
        numpy.testing.assert_almost_equal(
            evalhyd.evalp(_obs, _prd, ["QS"], t_msk=msk)[0],
            evalhyd.evalp(_obs[..., 99:], _prd[..., 99:], ["QS"])[0]
        )

    def test_conditions(self):
        with self.subTest(condtions="observed streamflow values"):
            cdt = numpy.array([["q_obs{<2000,>3000}"]], dtype='|S32')

            msk = (_obs[0] < 2000) | (_obs[0] > 3000)

            obs = _obs[..., msk]
            prd = _prd[..., msk]

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )

        with self.subTest(condtions="predicted streamflow statistics"):
            cdt = numpy.array([["q_prd_median{<=quantile0.7}"]], dtype='|S32')

            median = numpy.squeeze(numpy.median(_prd, 2))
            msk = median <= numpy.quantile(median, 0.7)

            obs = _obs[..., msk]
            prd = _prd[..., msk]

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )


class TestMissingData(unittest.TestCase):

    def test_nan(self):
        thr = numpy.array([[690, 534, 445, numpy.nan]])
        for metric in ("BS", "BSS", "BS_CRD", "BS_LBD", "QS", "CRPS"):
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    # missing data flagged as NaN
                    evalhyd.evalp(
                        [[4.7, numpy.nan, 5.5, 2.7, 4.1]],
                        [[[[5.3, 4.2, 5.7, 2.3, numpy.nan],
                           [4.3, 4.2, 4.7, 4.3, numpy.nan],
                           [5.3, 5.2, 5.7, 2.3, numpy.nan]]]],
                        [metric],
                        thr
                    )[0],
                    # missing data pairwise deleted from series
                    evalhyd.evalp(
                        [[4.7, 5.5, 2.7]],
                        [[[[5.3, 5.7, 2.3],
                           [4.3, 4.7, 4.3],
                           [5.3, 5.7, 2.3]]]],
                        [metric],
                        thr
                    )[0]
                )


class TestUncertainty(unittest.TestCase):

    def test_bootstrap(self):
        thr = numpy.array([[690, 534, 445, numpy.nan]])

        prd_1yr = numpy.genfromtxt(
            "./data/q_prd_1yr.csv", delimiter=',', skip_header=1
        )
        obs_1yr = numpy.genfromtxt(
            "./data/q_obs_1yr.csv", delimiter=',', skip_header=1
        )
        dts_1yr = numpy.genfromtxt(
            "./data/q_obs_1yr.csv", delimiter=',', dtype=str, skip_footer=1
        )

        obs_3yrs = numpy.hstack((obs_1yr,) * 3)
        prd_3yrs = numpy.hstack((prd_1yr,) * 3)

        for metric in ("BS", "BSS", "BS_CRD", "BS_LBD", "QS", "CRPS"):
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    # bootstrap with only one year of data
                    # (compare last sample only to have matching dimensions)
                    evalhyd.evalp(
                        obs_1yr[numpy.newaxis],
                        prd_1yr[numpy.newaxis, numpy.newaxis],
                        [metric],
                        q_thr=thr,
                        bootstrap={
                            "n_samples": 10, "len_sample": 3, "summary": 0
                        },
                        dts=dts_1yr
                    )[0][:, :, :, [0]],
                    # repeat year of data three times to correspond to a
                    # bootstrap sample of length 3
                    evalhyd.evalp(
                        obs_3yrs[numpy.newaxis],
                        prd_3yrs[numpy.newaxis, numpy.newaxis],
                        [metric],
                        q_thr=thr
                    )[0]
                )


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestMetrics)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestDecomposition)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestMasking)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestMissingData)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestUncertainty)
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    if not result.wasSuccessful():
        exit(1)
