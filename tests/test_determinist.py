import unittest
import numpy

import evalhyd


# load some predicted and observed streamflow
_prd = numpy.genfromtxt("./data/q_prd.csv", delimiter=',')[:5, :]
_obs = numpy.genfromtxt("./data/q_obs.csv", delimiter=',')[numpy.newaxis, :]


class TestMetrics(unittest.TestCase):

    expected = {
        'RMSE':
            [[[777.03427238]],
             [[776.87847854]],
             [[777.80021654]],
             [[778.15108180]],
             [[778.61486998]]],
        'NSE':
            [[[0.71891219]],
             [[0.71902490]],
             [[0.71835777]],
             [[0.71810361]],
             [[0.71776748]]],
        'KGE':
            [[[0.74808767]],
             [[0.74610620]],
             [[0.74411103]],
             [[0.74301085]],
             [[0.74176777]]],
        'KGEPRIME':
            [[[0.81314075]],
             [[0.81277485]],
             [[0.81203242]],
             [[0.81178671]],
             [[0.81138658]]]
    }

    def test_metrics_2d(self):
        for metric in self.expected.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, [metric])[0],
                    self.expected[metric]
                )

    def test_metrics_1d(self):
        for metric in self.expected.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs[0], _prd[0], [metric])[0],
                    [self.expected[metric][0]]
                )


class TestTransform(unittest.TestCase):

    def test_transform_sqrt(self):
        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"], "sqrt")[0],
            evalhyd.evald(_obs ** 0.5, _prd ** 0.5, ["NSE"])[0]
        )

    def test_transform_inv(self):
        eps = 0.01 * numpy.mean(_obs)
        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"], "inv")[0],
            evalhyd.evald(1 / (_obs + eps), 1 / (_prd + eps), ["NSE"])[0]
        )

    def test_transform_log(self):
        eps = 0.01 * numpy.mean(_obs)
        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"], "log")[0],
            evalhyd.evald(numpy.log(_obs + eps), numpy.log(_prd + eps),
                          ["NSE"])[0]
        )

    def test_transform_pow(self):
        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"], "pow", exponent=0.3)[0],
            evalhyd.evald(_obs ** 0.3, _prd ** 0.3, ["NSE"])[0]
        )


class TestMasking(unittest.TestCase):

    def test_masks(self):
        msk = numpy.ones(_obs.shape, dtype=bool)
        msk[..., :99] = False
        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"], t_msk=msk)[0],
            evalhyd.evald(_obs[..., 99:], _prd[..., 99:], ["NSE"])[0]
        )

    def test_conditions(self):
        with self.subTest(condtions="observed streamflow values"):
            cdt = numpy.array(["q_obs{<2000,>3000}"], dtype='|S32')

            msk = (_obs[0] < 2000) | (_obs[0] > 3000)

            obs = _obs[..., msk]
            prd = _prd[..., msk]

            numpy.testing.assert_almost_equal(
                evalhyd.evald(_obs, _prd, ["NSE"], m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"])[0]
            )

        with self.subTest(condtions="observed streamflow statistics"):
            cdt = numpy.array(["q_obs{>=median}"], dtype='|S32')

            msk = _obs[0] >= numpy.median(_obs)

            obs = _obs[..., msk]
            prd = _prd[..., msk]

            numpy.testing.assert_almost_equal(
                evalhyd.evald(_obs, _prd, ["NSE"], m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"])[0]
            )


class TestMissingData(unittest.TestCase):

    def test_nan(self):
        for metric in ('RMSE', 'NSE', 'KGE', 'KGEPRIME'):
            obs = numpy.array(
                [[4.7, numpy.nan, 5.5, 2.7, 4.1]]
            )
            prd = numpy.array(
                [[5.3, 4.2, 5.7, 2.3, numpy.nan],
                 [numpy.nan, 4.2, 4.7, 4.3, 3.3],
                 [5.3, 5.2, 5.7, numpy.nan, 3.9]]
            )

            with self.subTest(metric=metric):
                res = evalhyd.evald(obs, prd, [metric])[0]

                for i in range(prd.shape[0]):
                    msk = ~numpy.isnan(obs[0]) & ~numpy.isnan(prd[i])

                    numpy.testing.assert_almost_equal(
                        # missing data flagged as NaN
                        res[[i]],
                        # missing data pairwise deleted from series
                        evalhyd.evald(
                            obs[:, msk],
                            prd[i, msk][numpy.newaxis],
                            [metric]
                        )[0]
                    )


class TestUncertainty(unittest.TestCase):

    def test_bootstrap(self):
        prd_1yr = numpy.genfromtxt(
            "./data/q_prd_1yr.csv", delimiter=',', skip_header=1
        )
        obs_1yr = numpy.genfromtxt(
            "./data/q_obs_1yr.csv", delimiter=',', skip_header=1
        )[numpy.newaxis]
        dts_1yr = numpy.genfromtxt(
            "./data/q_obs_1yr.csv", delimiter=',', dtype=str, skip_footer=1
        )

        obs_3yrs = numpy.hstack((obs_1yr,) * 3)
        prd_3yrs = numpy.hstack((prd_1yr,) * 3)

        for metric in ('RMSE', 'NSE', 'KGE', 'KGEPRIME'):
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    # bootstrap with only one year of data
                    # (compare last sample only to have matching dimensions)
                    evalhyd.evald(
                        obs_1yr, prd_1yr, [metric],
                        bootstrap={
                            "n_samples": 10, "len_sample": 3, "summary": 0
                        },
                        dts=dts_1yr
                    )[0][..., [0]],
                    # repeat year of data three times to correspond to a
                    # bootstrap sample of length 3
                    evalhyd.evald(obs_3yrs, prd_3yrs, [metric])[0]
                )


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestMetrics)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestTransform)
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
