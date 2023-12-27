import unittest
import numpy

import evalhyd


# load some predicted and observed streamflow
_prd = (
    numpy.genfromtxt("./data/q_prd.csv", delimiter=',')
    [numpy.newaxis, numpy.newaxis, ...]
)
_obs = numpy.genfromtxt("./data/q_obs.csv", delimiter=',')[numpy.newaxis, :]

# list all available probabilistic metrics
_all_metrics = (
    # threshold-based
    'BS', 'BSS', 'BS_CRD', 'BS_LBD', 'REL_DIAG', 'CRPS_FROM_BS',
    # CDF-based
    'CRPS_FROM_ECDF',
    # quantile-based
    'QS', 'CRPS_FROM_QS',
    # contingency table-based
    'CONT_TBL', 'POD', 'POFD', 'FAR', 'CSI', 'ROCSS',
    # ranks-based
    'RANK_HIST', 'DS', 'AS',
    # intervals
    'CR', 'AW', 'AWN', 'WS',
    # multivariate
    'ES'
)

# list all available deterministic diagnostics
_all_diags = (
    'completeness'
)


class TestMetrics(unittest.TestCase):

    expected_thr = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('BS', 'BSS', 'BS_CRD', 'BS_LBD', 'REL_DIAG', 'CRPS_FROM_BS')
    }
    # /!\ stacked-up thresholds in CSV file for REL_DIAG
    #     because 7D metric so need to reshape array
    expected_thr['REL_DIAG'] = (
        expected_thr['REL_DIAG'].reshape(expected_thr['BS'].shape
                                         + (_prd.shape[2] + 1, 3))
    )

    expected_cdf = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('CRPS_FROM_ECDF',)
    }

    expected_qtl = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('QS', 'CRPS_FROM_QS')
    }

    expected_ct = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('CONT_TBL', 'POD', 'POFD', 'FAR', 'CSI', 'ROCSS')
    }
    # /!\ stacked-up thresholds in CSV file for CONT_TBL
    #     because 7D metric so need to reshape array
    expected_ct['CONT_TBL'] = (
        expected_ct['CONT_TBL'].reshape(expected_ct['POD'].shape + (4,))
    )

    expected_rk = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('RANK_HIST', 'DS', 'AS')
    }

    expected_itv = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('CR', 'AW', 'AWN', 'WS')
    }

    expected_mvr = {
        metric: (
            numpy.genfromtxt(f"./expected/evalp/{metric}.csv", delimiter=',')
            [numpy.newaxis, numpy.newaxis, numpy.newaxis, numpy.newaxis, ...]
        ) for metric in ('ES',)
    }

    def test_thresholds_metrics(self):
        thr = numpy.array([[690, 534, 445, numpy.nan]])
        for metric in self.expected_thr.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric], thr, "high")[0],
                    self.expected_thr[metric]
                )

    def test_cdf_metrics(self):
        for metric in self.expected_cdf.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric])[0],
                    self.expected_cdf[metric]
                )

    def test_quantiles_metrics(self):
        for metric in self.expected_qtl.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric])[0],
                    self.expected_qtl[metric]
                )

    def test_contingency_table_metrics(self):
        for metric in self.expected_ct.keys():
            thr = numpy.array([[690, 534, 445, numpy.nan]])
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric], thr, "low")[0],
                    self.expected_ct[metric]
                )

    def test_ranks_metrics(self):
        for metric in self.expected_rk.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric], seed=7)[0],
                    self.expected_rk[metric]
                )

    def test_intervals_metrics(self):
        lvl = numpy.array([30., 80.])
        for metric in self.expected_itv.keys():

            numpy.set_printoptions(precision=13)
            m = evalhyd.evalp(_obs, _prd, [metric], c_lvl=lvl)[0][0, 0, 0]
            numpy.savetxt(f"./expected/evalp/{metric}.csv", m, delimiter=',', fmt="%.13f")

            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(_obs, _prd, [metric], c_lvl=lvl)[0],
                    self.expected_itv[metric]
                )

    def test_multivariate_metrics(self):
        n_sit = 5

        multi_obs = numpy.repeat(_obs, repeats=n_sit, axis=0)
        multi_prd = numpy.repeat(_prd, repeats=n_sit, axis=0)

        for metric in self.expected_mvr.keys():
            with self.subTest(metric=metric):
                numpy.testing.assert_almost_equal(
                    evalhyd.evalp(multi_obs, multi_prd, [metric], seed=7)[0],
                    self.expected_mvr[metric]
                )


class TestDecomposition(unittest.TestCase):

    def test_brier_calibration_refinement(self):
        thr = numpy.array([[690, 534, 445]])
        bs, = evalhyd.evalp(_obs, _prd, ["BS"], thr, "high")
        bs_crd, = evalhyd.evalp(_obs, _prd, ["BS_CRD"], thr, "high")
        numpy.testing.assert_almost_equal(
            bs, bs_crd[..., 0] - bs_crd[..., 1] + bs_crd[..., 2]
        )

    def test_brier_likelihood_base_rate(self):
        thr = numpy.array([[690, 534, 445]])
        bs, = evalhyd.evalp(_obs, _prd, ["BS"], thr, "high")
        bs_lbd, = evalhyd.evalp(_obs, _prd, ["BS_LBD"], thr, "high")
        numpy.testing.assert_almost_equal(
            bs, bs_lbd[..., 0] - bs_lbd[..., 1] + bs_lbd[..., 2]
        )


class TestMasking(unittest.TestCase):

    def test_masks(self):
        msk = numpy.ones((_prd.shape[0], _prd.shape[1], 1, _prd.shape[3]),
                         dtype=bool)
        msk[..., :99] = False

        # TODO: figure out why passing views would not work
        obs = _obs[..., 99:].copy()
        prd = _prd[..., 99:].copy()

        numpy.testing.assert_almost_equal(
            evalhyd.evalp(_obs, _prd, ["QS"], t_msk=msk)[0],
            evalhyd.evalp(obs, prd, ["QS"])[0]
        )

    def test_conditions(self):
        with self.subTest(conditions="observed streamflow values"):
            cdt = numpy.array([["q_obs{<2000,>3000}"]])

            msk = (_obs[0] < 2000) | (_obs[0] > 3000)

            # TODO: figure out why passing views would not work
            obs = _obs[..., msk].copy()
            prd = _prd[..., msk].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )

        with self.subTest(conditions="predicted streamflow statistics 1"):
            cdt = numpy.array([["q_prd_median{<=qtl0.7}"]], dtype='|S32')

            median = numpy.squeeze(numpy.median(_prd, 2))
            msk = median <= numpy.quantile(median, 0.7)

            # TODO: figure out why passing views would not work
            obs = _obs[..., msk].copy()
            prd = _prd[..., msk].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )

        with self.subTest(conditions="predicted streamflow statistics 2"):
            cdt = numpy.array([["q_prd_median{>qtl0.3,<=qtl0.7}"]], dtype='|S32')

            median = numpy.squeeze(numpy.median(_prd, 2))
            msk = (
                    (median > numpy.quantile(median, 0.3))
                    & (median <= numpy.quantile(median, 0.7))
            )

            # TODO: figure out why passing views would not work
            obs = _obs[..., msk].copy()
            prd = _prd[..., msk].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )

        with self.subTest(conditions="time indices"):
            cdt = numpy.array([["t{20:80,80,81,82,83:311}"]],
                              dtype='|S32')

            # TODO: figure out why passing views would not work
            obs = _obs[..., 20:].copy()
            prd = _prd[..., 20:].copy()

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(_obs, _prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )

        with self.subTest(conditions="no subset"):
            cdt = numpy.array([["t{:}"]],
                              dtype='|S32')

            numpy.testing.assert_almost_equal(
                evalhyd.evalp(obs, prd, ["QS"], m_cdt=cdt)[0],
                evalhyd.evalp(obs, prd, ["QS"])[0]
            )


class TestMissingData(unittest.TestCase):

    def test_nan(self):
        thr = numpy.array([[690, 534, 445, numpy.nan]])
        for metric in _all_metrics:
            # skip ranks-based metrics because they contain a random element
            if metric in ("RANK_HIST", "DS", "AS"):
                continue

            with self.subTest(metric=metric):
                lvl = numpy.array([30., 80.])
                numpy.testing.assert_almost_equal(
                    # missing data flagged as NaN
                    evalhyd.evalp(
                        numpy.array([[4.7, numpy.nan, 5.5, 2.7, 4.1]]),
                        numpy.array([[[[5.3, 4.2, 5.7, 2.3, numpy.nan],
                                       [4.3, 4.2, 4.7, 4.3, numpy.nan],
                                       [5.3, 5.2, 5.7, 2.3, numpy.nan]]]]),
                        [metric],
                        thr,
                        "high",
                        lvl
                    )[0],
                    # missing data pairwise deleted from series
                    evalhyd.evalp(
                        numpy.array([[4.7, 5.5, 2.7]]),
                        numpy.array([[[[5.3, 5.7, 2.3],
                                       [4.3, 4.7, 4.3],
                                       [5.3, 5.7, 2.3]]]]),
                        [metric],
                        thr,
                        "high",
                        lvl
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

        for metric in _all_metrics:
            # skip ranks-based metrics because they contain a random element
            if metric in ("RANK_HIST", "DS", "AS"):
                continue

            with self.subTest(metric=metric):
                lvl = numpy.array([30., 80.])
                numpy.testing.assert_almost_equal(
                    # bootstrap with only one year of data
                    # (compare last sample only to have matching dimensions)
                    evalhyd.evalp(
                        obs_1yr[numpy.newaxis],
                        prd_1yr[numpy.newaxis, numpy.newaxis],
                        [metric],
                        q_thr=thr,
                        events="high",
                        bootstrap={
                            "n_samples": 10, "len_sample": 3, "summary": 0
                        },
                        dts=dts_1yr,
                        c_lvl=lvl
                    )[0][:, :, :, [0]],
                    # repeat year of data three times to correspond to a
                    # bootstrap sample of length 3
                    evalhyd.evalp(
                        obs_3yrs[numpy.newaxis],
                        prd_3yrs[numpy.newaxis, numpy.newaxis],
                        [metric],
                        q_thr=thr,
                        events="high",
                        c_lvl=lvl
                    )[0]
                )


class TestMultiDimensional(unittest.TestCase):

    thr = numpy.array([[690, 534, 445, numpy.nan]])
    events = "high"
    lvl = numpy.array([30., 80.])
    seed = 7

    # skip ranks-based metrics because they contain a random element
    metrics = [m for m in _all_metrics if m not in ("RANK_HIST", "DS", "AS")]

    def test_multi_sites(self):
        n_sit = 3
        multi_obs = numpy.repeat(_obs, repeats=n_sit, axis=0)
        multi_prd = numpy.repeat(_prd, repeats=n_sit, axis=0)
        multi_thr = numpy.repeat(self.thr, repeats=n_sit, axis=0)

        # skip multisite metrics because their result is not the sum of sites
        metrics = [m for m in self.metrics if m not in ("ES",)]

        multi = evalhyd.evalp(
            multi_obs,
            multi_prd,
            metrics,
            q_thr=multi_thr,
            events=self.events,
            c_lvl=self.lvl,
            seed=self.seed
        )

        mono = evalhyd.evalp(
            _obs,
            _prd,
            metrics,
            q_thr=self.thr,
            events=self.events,
            c_lvl=self.lvl,
            seed=self.seed
        )

        for m, metric in enumerate(metrics):
            for site in range(n_sit):
                with self.subTest(metric=metric, site=site):
                    numpy.testing.assert_almost_equal(
                        multi[m][[site]], mono[m]
                    )

    def test_multi_leadtimes(self):
        n_ldt = 7
        multi_prd = numpy.repeat(_prd, repeats=n_ldt, axis=1)

        multi = evalhyd.evalp(
            _obs,
            multi_prd,
            self.metrics,
            q_thr=self.thr,
            events=self.events,
            c_lvl=self.lvl,
            seed=self.seed
        )

        mono = evalhyd.evalp(
            _obs,
            _prd,
            self.metrics,
            q_thr=self.thr,
            events=self.events,
            c_lvl=self.lvl,
            seed=self.seed
        )

        for m, metric in enumerate(self.metrics):
            for leadtime in range(n_ldt):
                with self.subTest(metric=metric, leadtime=leadtime):
                    numpy.testing.assert_almost_equal(
                        multi[m][:, [leadtime]], mono[m]
                    )

    def test_multi_sites_multi_leadtimes(self):
        n_sit = 3
        n_ldt = 7

        multi_obs = numpy.repeat(_obs, repeats=n_sit, axis=0)
        multi_obs += numpy.random.randint(
            low=0, high=200, size=(n_sit, multi_obs.shape[1])
        )

        multi_prd = numpy.repeat(_prd, repeats=n_sit, axis=0)
        multi_prd = numpy.repeat(multi_prd, repeats=n_ldt, axis=1)
        multi_prd += numpy.random.randint(
            low=0, high=200, size=(n_sit, n_ldt, *multi_prd.shape[2:])
        )

        multi_thr = numpy.repeat(self.thr, repeats=n_sit, axis=0)

        # skip multisite metrics because their result is not the sum of sites
        metrics = [m for m in self.metrics if m not in ("ES",)]

        multi = evalhyd.evalp(
            multi_obs,
            multi_prd,
            metrics,
            q_thr=multi_thr,
            events=self.events,
            c_lvl=self.lvl,
            seed=self.seed
        )

        for m, metric in enumerate(metrics):
            for sit in range(n_sit):
                for ldt in range(n_ldt):

                    mono = evalhyd.evalp(
                        multi_obs[[sit]],
                        multi_prd[[sit]][:, [ldt]],
                        [metric],
                        q_thr=self.thr,
                        events=self.events,
                        c_lvl=self.lvl,
                        seed=self.seed
                    )

                    with self.subTest(metric=metric, site=sit, leadtime=ldt):
                        numpy.testing.assert_almost_equal(
                            multi[m][sit, ldt], mono[0][0, 0]
                        )


class TestDiagnostics(unittest.TestCase):

    def test_completeness(self):
        obs = numpy.array(
            [[4.7, 4.3, numpy.nan, 2.7, 4.1, 5.0]]
        )

        prd = numpy.array(
            [[[[5.3, numpy.nan, 5.7, 2.3, 3.3, numpy.nan],
               [4.3, numpy.nan, 4.7, 4.3, 3.4, numpy.nan],
               [5.3, numpy.nan, 5.7, 2.3, 3.8, numpy.nan]],
              [[numpy.nan, 4.2, 5.7, 2.3, 3.1, 4.1],
               [numpy.nan, 4.2, 4.7, 4.3, 3.3, 2.8],
               [numpy.nan, 5.2, 5.7, 2.3, 3.9, 3.5]]]]
        )

        msk = numpy.array(
            [[[[True, True, True, False, True, True],
               [True, True, True, True, True, True]],
              [[True, True, True, True, True, False],
               [True, True, True, True, True, True]]]]
        )

        exp = numpy.array(
            [[[[2.],
               [3.]],
              [[3.],
               [4.]]]]
        )

        numpy.testing.assert_almost_equal(
            exp,
            evalhyd.evalp(
                obs, prd, ["QS"], t_msk=msk, diagnostics=["completeness"]
            )[1]
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
