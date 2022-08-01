import unittest
import numpy

import evalhyd


# load some predicted and observed streamflow
_prd = numpy.genfromtxt("./data/q_prd.csv", delimiter=',')[:5, :]
_obs = numpy.genfromtxt("./data/q_obs.csv", delimiter=',')[numpy.newaxis, :]


class TestMetrics(unittest.TestCase):

    expected = {
        'RMSE':
            [[777.03427238],
             [776.87847854],
             [777.80021654],
             [778.15108180],
             [778.61486998]],
        'NSE':
            [[0.71891219],
             [0.71902490],
             [0.71835777],
             [0.71810361],
             [0.71776748]],
        'KGE':
            [[0.74808767],
             [0.74610620],
             [0.74411103],
             [0.74301085],
             [0.74176777]],
        'KGEPRIME':
            [[0.81314075],
             [0.81277485],
             [0.81203242],
             [0.81178671],
             [0.81138658]]
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
                    self.expected[metric][0]
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


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestMetrics)
    )
    test_suite.addTests(
        test_loader.loadTestsFromTestCase(TestTransform)
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    if not result.wasSuccessful():
        exit(1)
