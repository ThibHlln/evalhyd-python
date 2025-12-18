import unittest
import numpy

import evalhyd


# load some predicted and observed streamflow
_prd = numpy.genfromtxt("./data/q_prd.csv", delimiter=',')[:, :]
_obs = numpy.genfromtxt("./data/q_obs.csv", delimiter=',')[numpy.newaxis, :]

_thr = numpy.repeat(
    numpy.array([[690, 534, 445, numpy.nan]]), repeats=_prd.shape[0], axis=0
)
_events = "high"

# list all available deterministic metrics
_all_metrics = (
    # errors-based
    'MAE', 'MARE', 'MSE', 'RMSE',
    # efficiencies-based
    'NSE', 'KGE', 'KGE_D', 'KGEPRIME', 'KGEPRIME_D', 'KGENP', 'KGENP_D',
    # contingency table-based
    'CONT_TBL'
)

# list all available deterministic diagnostics
_all_diags = (
    'completeness'
)


class TestMetrics(unittest.TestCase):

    expected = {
        metric: (
            numpy.genfromtxt(f"./expected/evald/{metric}.csv", delimiter=',')
            [:, numpy.newaxis, numpy.newaxis]
        ) for metric in _all_metrics
    }
    # /!\ stacked-up thresholds in CSV file for CONT_TBL
    #     because 5D metric so need to reshape array
    expected['CONT_TBL'] = (
        expected['CONT_TBL'].reshape(expected['NSE'].shape + (4, 4))
    )

    def test_metrics_2d(self):
        for metric in self.expected.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, [metric],
                                  q_thr=_thr, events=_events)[0],
                    self.expected[metric]
                )

    def test_metrics_1d(self):
        for metric in self.expected.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs[0], _prd[0], [metric],
                                  q_thr=_thr[0], events=_events)[0],
                    [self.expected[metric][0]]
                )


class TestTransform(unittest.TestCase):

    def test_transform_sqrt(self):
        for metric in _all_metrics:
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, ["NSE"], transform="sqrt",
                                  q_thr=_thr, events=_events)[0],
                    evalhyd.evald(_obs ** 0.5, _prd ** 0.5, ["NSE"],
                                  q_thr=_thr ** 0.5, events=_events)[0]
                )

    def test_transform_inv(self):
        eps = 0.01 * numpy.mean(_obs)
        for metric in _all_metrics:
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, ["NSE"], transform="inv",
                                  q_thr=_thr, events=_events)[0],
                    evalhyd.evald(1 / (_obs + eps), 1 / (_prd + eps), ["NSE"],
                                  q_thr=1 / (_thr + eps), events=_events)[0]
                )

    def test_transform_log(self):
        eps = 0.01 * numpy.mean(_obs)
        for metric in _all_metrics:
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, ["NSE"], transform="log",
                                  q_thr=_thr, events=_events)[0],
                    evalhyd.evald(numpy.log(_obs + eps), numpy.log(_prd + eps),
                                  ["NSE"],
                                  q_thr=numpy.log(_thr + eps), events=_events)[0]
                )

    def test_transform_pow(self):
        for metric in _all_metrics:
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evald(_obs, _prd, ["NSE"],
                                  q_thr=_thr, events=_events,
                                  transform="pow", exponent=0.3)[0],
                    evalhyd.evald(_obs ** 0.3, _prd ** 0.3, ["NSE"],
                                  q_thr=_thr ** 0.3, events=_events)[0]
                )


class TestMasking(unittest.TestCase):

    def test_masks(self):
        msk = numpy.ones(_prd.shape, dtype=bool)
        msk = msk[:, numpy.newaxis, :]
        msk[..., :99] = False

        # TODO: figure out why passing views would not work
        obs = _obs[..., 99:].copy()
        prd = _prd[..., 99:].copy()

        numpy.testing.assert_almost_equal(
            evalhyd.evald(_obs, _prd, ["NSE"],
                          q_thr=_thr, events=_events, t_msk=msk)[0],
            evalhyd.evald(obs, prd, ["NSE"],
                          q_thr=_thr, events=_events)[0]
        )

    def test_conditions(self):
        with self.subTest(conditions="observed streamflow values"):
            cdt = numpy.array([["q_obs{<2000,>3000}"]] * _prd.shape[0],
                              dtype='|S32')

            msk = (_obs[0] < 2000) | (_obs[0] > 3000)

            # TODO: figure out why passing views would not work
            obs = _obs[..., msk].copy()
            prd = _prd[..., msk].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evald(_obs, _prd, ["NSE"],
                              q_thr=_thr, events=_events, m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"],
                              q_thr=_thr, events=_events)[0]
            )

        with self.subTest(conditions="observed streamflow statistics"):
            cdt = numpy.array([["q_obs{>=median}"]] * _prd.shape[0],
                              dtype='|S32')

            msk = _obs[0] >= numpy.median(_obs)

            # TODO: figure out why passing views would not work
            obs = _obs[..., msk].copy()
            prd = _prd[..., msk].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evald(_obs, _prd, ["NSE"],
                              q_thr=_thr, events=_events, m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"],
                              q_thr=_thr, events=_events)[0]
            )

        with self.subTest(conditions="time indices"):
            cdt = numpy.array([["t{20:311}"]] * (_prd.shape[0] - 4) +
                              [["t{20:100,100:311}"],
                               ["t{20,21,22,23,24:311}"],
                               ["t{20,21,22,23:309,309,310}"],
                               ["t{20:80,80,81,82,83:311}"]],
                              dtype='|S32')

            # TODO: figure out why passing views would not work
            obs = _obs[..., 20:].copy()
            prd = _prd[..., 20:].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evald(_obs, _prd, ["NSE"],
                              q_thr=_thr, events=_events, m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"],
                              q_thr=_thr, events=_events)[0]
            )

        with self.subTest(conditions="no subset"):
            cdt = numpy.array([["t{:}"]] * _prd.shape[0],
                              dtype='|S32')

            numpy.testing.assert_almost_equal(
                evalhyd.evald(obs, prd, ["NSE"],
                              q_thr=_thr, events=_events, m_cdt=cdt)[0],
                evalhyd.evald(obs, prd, ["NSE"],
                              q_thr=_thr, events=_events)[0]
            )


class TestMissingData(unittest.TestCase):

    def test_nan(self):
        for metric in _all_metrics:
            obs = numpy.array(
                [[4.7, numpy.nan, 5.5, 2.7, 4.1]]
            )
            prd = numpy.array(
                [[5.3, 4.2, 5.7, 2.3, numpy.nan],
                 [numpy.nan, 4.2, 4.7, 4.3, 3.3],
                 [5.3, 5.2, 5.7, numpy.nan, 3.9]]
            )
            thr = numpy.array(
                [[4., 5.],
                 [4., 5.],
                 [4., 5.]]
            )
            events = "low"

            with self.subTest(metric=metric):
                res = evalhyd.evald(obs, prd, [metric],
                                    q_thr=thr, events=events)[0]

                for i in range(prd.shape[0]):
                    msk = ~numpy.isnan(obs[0]) & ~numpy.isnan(prd[i])

                    numpy.testing.assert_almost_equal(
                        # missing data flagged as NaN
                        res[[i]],
                        # missing data pairwise deleted from series
                        evalhyd.evald(
                            obs[:, msk],
                            prd[i, msk][numpy.newaxis],
                            [metric],
                            q_thr=thr[i, :][numpy.newaxis], events=events
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

        thr = numpy.repeat(
            numpy.array([[690, 534, 445, numpy.nan]]),
            repeats=prd_1yr.shape[0], axis=0
        )
        events = "low"

        for metric in _all_metrics:
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    # bootstrap with only one year of data
                    # (compare last sample only to have matching dimensions)
                    evalhyd.evald(
                        obs_1yr, prd_1yr, [metric],
                        q_thr=thr,
                        events=events,
                        bootstrap={
                            "n_samples": 10, "len_sample": 3, "summary": 0
                        },
                        dts=dts_1yr
                    )[0][:, :, [0]],
                    # repeat year of data three times to correspond to a
                    # bootstrap sample of length 3
                    evalhyd.evald(obs_3yrs, prd_3yrs, [metric],
                                  q_thr=thr, events=events)[0]
                )


class TestDiagnostics(unittest.TestCase):

    def test_completeness(self):
        obs = numpy.array(
            [[4.7, 4.3, numpy.nan, 2.7, 4.1, 5.0]]
        )

        prd = numpy.array(
            [[5.3, numpy.nan, 5.7, 2.3, 3.3, 4.1],
             [4.3, 4.2, 4.7, 4.3, 3.3, 2.8],
             [5.3, numpy.nan, 5.7, 2.3, 3.8, numpy.nan]]
        )

        msk = numpy.array(
            [[[True, True, True, False, True, True],
              [True, True, True, True, True, True]],
             [[True, True, True, True, True, False],
              [True, True, True, True, True, True]],
             [[True, True, True, False, False, True],
              [True, True, True, True, True, True]]]
        )

        exp = numpy.array(
            [[[3.],
              [4.]],
             [[4.],
              [5.]],
             [[1.],
              [3.]]]
        )

        numpy.testing.assert_almost_equal(
            exp,
            evalhyd.evald(
                obs, prd, ["NSE"], t_msk=msk, diagnostics=["completeness"]
            )[1]
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
