#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/evald.hpp"
#include "evalhyd/evalp.hpp"

namespace py = pybind11;

// Python Module and Docstrings
PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Utility for evaluation of streamflow predictions.
    )pbdoc";

    // deterministic evaluation
    m.def(
        "evald", evalhyd::evald<xt::pytensor<double, 1>>,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    1D array of streamflow observations.
                    shape: (time,)

                q_prd: `numpy.ndarray`
                    1D array of streamflow predictions.
                    shape: (time,)

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.
                    
                transform: `str`, optional
                   The transformation to apply to both streamflow observations
                   and predictions prior to the calculation of the *metrics*.
                   The options are listed in the table below.

                   ========================  ==================================
                   transformations           details
                   ========================  ==================================
                   ``"sqrt"``                The square root function
                                             **f(Q) = √Q** is applied.
                   ``"pow"``                 The power function
                                             **f(Q) = Qⁿ** is applied (where
                                             the power **n** can be set through
                                             the *exponent* parameter).
                   ``"inv"``                 The reciprocal function
                                             **f(Q) = 1/Q** is applied.
                   ``"log"``                 The natural logarithm function
                                             **f(Q) = ln(Q)** is applied.
                   ========================  ==================================

                exponent: `float`, optional
                   The value of the exponent n to use when the *transform* is
                   the power function. If not provided (or set to default value
                   1), the streamflow observations and predictions remain
                   untransformed.

                epsilon: `float`, optional
                   The value of the small constant ε to add to both the
                   streamflow observations and predictions prior to the
                   calculation of the *metrics* when the *transform* is the
                   reciprocal function, the natural logarithm, or the power
                   function with a negative exponent (since none are defined
                   for 0). If not provided (or set to default value -9),
                   one hundredth of the mean of the streamflow observations
                   is used as value for epsilon, as recommended by
                   `Pushpalatha et al. (2012)
                   <https://doi.org/10.1016/j.jhydrol.2011.11.055>`_.

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.
                    shape: [(components,)+]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("transform")="none", py::arg("exponent")=1,
        py::arg("epsilon")=-9
    );
    m.def(
        "evald", evalhyd::evald<xt::pytensor<double, 2>>,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    2D array of streamflow observations.
                    shape: (1, time)

                q_prd: `numpy.ndarray`
                    2D array of streamflow predictions.
                    shape: (1+, time)

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.

                transform: `str`, optional
                   The transformation to apply to both streamflow observations
                   and predictions prior to the calculation of the *metrics*.
                   The options are listed in the table below.

                   ========================  ==================================
                   transformations           details
                   ========================  ==================================
                   ``"sqrt"``                The square root function
                                             **f(Q) = √Q** is applied.
                   ``"pow"``                 The power function
                                             **f(Q) = Qⁿ** is applied (where
                                             the power **n** can be set through
                                             the *exponent* parameter).
                   ``"inv"``                 The reciprocal function
                                             **f(Q) = 1/Q** is applied.
                   ``"log"``                 The natural logarithm function
                                             **f(Q) = ln(Q)** is applied.
                   ========================  ==================================

                exponent: `float`, optional
                   The value of the exponent n to use when the *transform* is
                   the power function. If not provided (or set to default value
                   1), the streamflow observations and predictions remain
                   untransformed.

                epsilon: `float`, optional
                   The value of the small constant ε to add to both the
                   streamflow observations and predictions prior to the
                   calculation of the *metrics* when the *transform* is the
                   reciprocal function, the natural logarithm, or the power
                   function with a negative exponent (since none are defined
                   for 0). If not provided (or set to default value -9),
                   one hundredth of the mean of the streamflow observations
                   is used as value for epsilon, as recommended by
                   `Pushpalatha et al. (2012)
                   <https://doi.org/10.1016/j.jhydrol.2011.11.055>`_.

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.
                    shape: [(1+, components), ...]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("transform")="none", py::arg("exponent")=1,
        py::arg("epsilon")=-9
    );

    // probabilistic evaluation
    py::list empty_1d;
    py::list empty_2d;
    empty_2d.append(py::list());

    m.def(
        "evalp", evalhyd::evalp,
        R"pbdoc(
            Function to evaluate probabilistic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                    2D array of streamflow observations.
                    shape: (sites, time)

                q_prd: `numpy.ndarray`
                    4D array of streamflow predictions.
                    shape: (sites, lead times, members, time)

                metrics: `List[str]`
                    The sequence of evaluation metrics to be computed.

                q_thr: `List[float]`, optional
                    The streamflow threshold(s) to consider for the *metrics*
                    assessing the prediction of exceedance events. If not
                    provided, set to default value as an empty `list`.
                    shape: (thresholds,)

                t_msk: `numpy.ndarray`, optional
                    2D array of masks to generate temporal subsets of the whole
                    streamflow time series (where True/False is used for the
                    time steps to include/discard in a given subset). If not
                    provided, no subset is performed and only one set of metrics
                    is returned corresponding to the whole time series. If
                    provided, as many sets of metrics are returned as they are
                    masks provided.
                    shape: (subsets, time)

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.
                    shape: [(sites, lead times, subsets, {quantiles,} {thresholds,} {components}), ...]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("q_thr") = empty_1d, py::arg("t_msk") = empty_2d
    );
}
