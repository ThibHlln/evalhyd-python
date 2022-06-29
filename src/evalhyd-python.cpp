#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/determinist.hpp"
#include "evalhyd/probabilist.hpp"

namespace py = pybind11;

// Python Module and Docstrings
PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Utility for evaluation of streamflow predictions.

        .. currentmodule:: evalhyd

        .. autosummary::

           evald
           evalp
    )pbdoc";

    // deterministic evaluation
    m.def(
        "evald", evalhyd::evald<xt::pytensor<double, 1>>,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    1D array of streamflow observations.

                q_prd: `numpy.ndarray`
                    1D array of streamflow predictions.

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.

            :Examples:

               >>> import numpy
               >>> import evalhyd
               >>> obs = numpy.array(
               ...     [4.7, 4.3, 5.5, 2.7, 4.1]
               ... )
               >>> prd = numpy.array(
               ...     [5.3, 4.2, 5.7, 2.3, 3.1]
               ... )

               >>> nse, = evalhyd.evalp(obs, prd, ['NSE'])
               >>> print(nse)
               [0.6254771]

        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics")
    );
    m.def(
        "evald", evalhyd::evald<xt::pytensor<double, 2>>,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    2D array of streamflow observations (with its temporal
                    dimension on axis 1).

                q_prd: `numpy.ndarray`
                    2D array of streamflow predictions (with its temporal
                    dimension on axis 1).

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.

            :Examples:

               >>> import numpy
               >>> import evalhyd
               >>> obs = numpy.array(
               ...     [[4.7, 4.3, 5.5, 2.7, 4.1]]
               ... )
               >>> prd = numpy.array(
               ...     [[5.3, 4.2, 5.7, 2.3, 3.1],
               ...      [4.3, 4.2, 4.7, 4.3, 3.3],
               ...      [5.3, 5.2, 5.7, 2.3, 3.9]]
               ... )

               >>> nse, = evalhyd.evalp(obs, prd, ['NSE'])
               >>> print(nse)
               [[0.6254771 ]
                [0.04341603]
                [0.66364504]]

        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics")
    );

    // probabilistic evaluation
    m.def(
        "evalp", evalhyd::evalp,
        R"pbdoc(
            Function to evaluate probabilistic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    2D array of streamflow observations (with size 1 for
                    axis 0, and with the temporal dimension on axis 1).

                q_prd: `numpy.ndarray`
                    2D array of streamflow predictions (with the ensemble
                    members on axis 0, and with the temporal dimension on
                    axis 1).

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.

                q_thr: `List[float]`, optional
                    The streamflow threshold(s) to consider for the *metrics*
                    assessing the prediction of exceedance events. If not
                    provided, set to default value as an empty `list`.

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.

            :Examples:

               >>> import numpy
               >>> import evalhyd
               >>> obs = numpy.array(
               ...     [[4.7, 4.3, 5.5, 2.7, 4.1]]
               ... )
               >>> prd = numpy.array(
               ...     [[5.3, 4.2, 5.7, 2.3, 3.1],
               ...      [4.3, 4.2, 4.7, 4.3, 3.3],
               ...      [5.3, 5.2, 5.7, 2.3, 3.9]]
               ... )

               >>> bs, bs_lbd = evalhyd.evalp(obs, prd, ['BS', 'BS_LBD'], [4., 5.])
               >>> print(bs)
               [[0.22222222]
                [0.13333333]]
               >>> print(bs_lbd)
               [[0.07222222 0.02777778 0.17777778]
                [0.07222222 0.02777778 0.08888889]]

               >>> crps, = evalhyd.evalp(obs, prd, ['CRPS'])
               >>> print(crps)
               [[0.24193548]]

        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("q_thr") = py::list()
    );
}
