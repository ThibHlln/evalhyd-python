#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define FORCE_IMPORT_ARRAY
#include <xtensor/xview.hpp>
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/evald.hpp"
#include "evalhyd/evalp.hpp"

namespace py = pybind11;
using namespace py::literals;

// reshape 1D tensors to 2D tensors
auto evald_1d(
    const xt::xtensor<double, 1>& q_obs,
    const xt::xtensor<double, 1>& q_prd,
    const std::vector<std::string>& metrics,
    const std::string& transform = "none",
    const double exponent = 1,
    double epsilon = -9,
    const xt::xtensor<bool, 2>& t_msk = {},
    const xt::xtensor<std::array<char, 32>, 1>& m_cdt = {},
    const std::unordered_map<std::string, int>& bootstrap =
        {{"n_samples", -9}, {"len_sample", -9}, {"summary", 0}},
    const std::vector<std::string>& dts = {}
)
{
    return evalhyd::evald(
        xt::view(q_obs, xt::newaxis(), xt::all()),
        xt::view(q_prd, xt::newaxis(), xt::all()),
        metrics,
        transform,
        exponent,
        epsilon,
        t_msk,
        m_cdt,
        bootstrap,
        dts
    );
}

// Python Module and Docstrings
PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Utility for evaluation of streamflow predictions.
    )pbdoc";

    // deterministic evaluation
    m.def(
        "evald", evald_1d,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                   1D array of streamflow observations. Time steps with
                   missing observations must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (time,)

                q_prd: `numpy.ndarray`
                   1D array of streamflow predictions. Time steps with
                   missing predictions must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (time,)

                metrics: `List[str]`
                   The sequence of evaluation metrics to be computed.

                transform: `str`, optional
                   The transformation to apply to both streamflow observations
                   and predictions prior to the calculation of the *metrics*.

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
                   is used as value for epsilon.

                t_msk: `numpy.ndarray`, optional
                   1D array of mask(s) used to generate temporal subsets of
                   the whole streamflow time series (where True/False is used for
                   the time steps to include/discard in a given subset). If not
                   provided and neither is *m_cdt*, no subset is performed. If
                   provided, masks must feature the same number of dimensions as
                   observations and predictions, and it must broadcastable with
                   both of them.
                   shape: (subsets, time)

                m_cdt: `numpy.ndarray`, optional
                   1D array of masking condition(s) to use to generate
                   temporal subsets. Each condition consists in a string and
                   can be specified on observed streamflow values/statistics
                   (mean, median, quantile), or on time indices. If provided
                   in combination with *t_msk*, the latter takes precedence.
                   If not provided and neither is *t_msk*, no subset is
                   performed. If provided, only one condition per time series
                   of observations can be provided.
                   shape: (subsets,)

                bootstrap: `dict`, optional
                   Parameters for the bootstrapping method used to estimate the
                   sampling uncertainty in the evaluation of the predictions.
                   Three parameters are mandatory ('n_samples' the number of
                   random samples, 'len_sample' the length of one sample in
                   number of years, and 'summary' the statistics to return to
                   characterise the sampling distribution), and one parameter
                   is optional ('seed'). If not provided, no bootstrapping is
                   performed. If provided, *dts* must also be provided.

                dts: `List[str]`, optional
                   Datetimes. The corresponding date and time for the temporal
                   dimension of the streamflow observations and predictions.
                   The date and time must be specified in a string following
                   the ISO 8601-1:2019 standard, i.e. "YYYY-MM-DD hh:mm:ss"
                   (e.g. the 21st of May 2007 at 4 in the afternoon is
                   "2007-05-21 16:00:00"). If provided, it is only used if
                   *bootstrap* is also provided.
                   shape: (time,)

            :Returns:

                `List[numpy.ndarray]`
                    The sequence of evaluation metrics computed
                    in the same order as given in *metrics*.
                    shape: [(1, subsets, samples), ...]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("transform") = "none",
        py::arg("exponent") = 1,
        py::arg("epsilon") = -9,
        py::arg("t_msk") = xt::pytensor<bool, 2>({0}),
        py::arg("m_cdt") = xt::pytensor<std::array<char, 32>, 1>({}),
        py::arg("bootstrap") =
            py::dict("n_samples"_a=-9, "len_sample"_a=-9, "summary"_a=0),
        py::arg("dts") = py::list()
    );
    m.def(
        "evald", evalhyd::evald,
        R"pbdoc(
            Function to evaluate deterministic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                   2D array of streamflow observations. Time steps with
                   missing observations must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (1, time)

                q_prd: `numpy.ndarray`
                   2D array of streamflow predictions. Time steps with
                   missing predictions must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (series, time)

                metrics: `List[str]`
                   The sequence of evaluation metrics to be computed.

                transform: `str`, optional
                   The transformation to apply to both streamflow observations
                   and predictions prior to the calculation of the *metrics*.

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
                   is used as value for epsilon.

                t_msk: `numpy.ndarray`, optional
                   2D array of mask(s) used to generate temporal subsets of
                   the whole streamflow time series (where True/False is used for
                   the time steps to include/discard in a given subset). If not
                   provided and neither is *m_cdt*, no subset is performed. If
                   provided, masks must feature the same number of dimensions as
                   observations and predictions, and it must broadcastable with
                   both of them.
                   shape: (subsets, time)

                m_cdt: `numpy.ndarray`, optional
                   1D array of masking condition(s) to use to generate
                   temporal subsets. Each condition consists in a string and
                   can be specified on observed streamflow values/statistics
                   (mean, median, quantile), or on time indices. If provided
                   in combination with *t_msk*, the latter takes precedence.
                   If not provided and neither is *t_msk*, no subset is
                   performed. If provided, only one condition per time series
                   of observations can be provided.
                   shape: (subsets,)

                bootstrap: `dict`, optional
                   Parameters for the bootstrapping method used to estimate the
                   sampling uncertainty in the evaluation of the predictions.
                   Three parameters are mandatory ('n_samples' the number of
                   random samples, 'len_sample' the length of one sample in
                   number of years, and 'summary' the statistics to return to
                   characterise the sampling distribution), and one parameter
                   is optional ('seed'). If not provided, no bootstrapping is
                   performed. If provided, *dts* must also be provided.

                dts: `List[str]`, optional
                   Datetimes. The corresponding date and time for the temporal
                   dimension of the streamflow observations and predictions.
                   The date and time must be specified in a string following
                   the ISO 8601-1:2019 standard, i.e. "YYYY-MM-DD hh:mm:ss"
                   (e.g. the 21st of May 2007 at 4 in the afternoon is
                   "2007-05-21 16:00:00"). If provided, it is only used if
                   *bootstrap* is also provided.
                   shape: (time,)

            :Returns:

                `List[numpy.ndarray]`
                   The sequence of evaluation metrics computed
                   in the same order as given in *metrics*.
                   shape: [(series, subsets, samples), ...]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("transform") = "none",
        py::arg("exponent") = 1,
        py::arg("epsilon") = -9,
        py::arg("t_msk") = xt::pytensor<bool, 2>({0}),
        py::arg("m_cdt") = xt::pytensor<std::array<char, 32>, 1>({}),
        py::arg("bootstrap") =
            py::dict("n_samples"_a=-9, "len_sample"_a=-9, "summary"_a=0),
        py::arg("dts") = py::list()
    );

    // probabilistic evaluation
    m.def(
        "evalp", evalhyd::evalp,
        R"pbdoc(
            Function to evaluate probabilistic streamflow predictions.

            :Parameters:

                q_obs: `numpy.ndarray`
                   2D array of streamflow observations. Time steps with
                   missing observations must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (sites, time)

                q_prd: `numpy.ndarray`
                   4D array of streamflow predictions. Time steps with
                   missing predictions must be assigned `numpy.nan`
                   values. Those time steps will be ignored both in
                   the observations and in the predictions before the
                   *metrics* are computed.
                   shape: (sites, lead times, members, time)

                metrics: `List[str]`
                   The sequence of evaluation metrics to be computed.

                q_thr: `List[float]`, optional
                   The streamflow threshold(s) to consider for the *metrics*
                   assessing the prediction of exceedance events. If not
                   provided, set to default value as an empty `list`.
                   shape: (thresholds,)

                t_msk: `numpy.ndarray`, optional
                   4D array of masks to generate temporal subsets of the whole
                   streamflow time series (where True/False is used for the
                   time steps to include/discard in a given subset). If not
                   provided, no subset is performed and only one set of metrics
                   is returned corresponding to the whole time series. If
                   provided, as many sets of metrics are returned as they are
                   masks provided.
                   shape: (sites, lead times, subsets, time)

                m_cdt: `numpy.ndarray`, optional
                   2D array of conditions to generate temporal subsets. Each
                   condition consists in a string and can be specified on
                   observed/predicted streamflow values/statistics (mean,
                   median, quantile), or on time indices. If provided in
                   combination with t_msk, the latter takes precedence. If not
                   provided and neither is t_msk, no subset is performed and
                   only one set of metrics is returned corresponding to the
                   whole time series. If provided, as many sets of metrics are
                   returned as they are conditions provided.
                   shape: (sites, subsets)

                bootstrap: `dict`, optional
                   Parameters for the bootstrapping method used to estimate the
                   sampling uncertainty in the evaluation of the predictions.
                   Three parameters are mandatory ('n_samples' the number of
                   random samples, 'len_sample' the length of one sample in
                   number of years, and 'summary' the statistics to return to
                   characterise the sampling distribution), and one parameter
                   is optional ('seed'). If not provided, no bootstrapping is
                   performed. If provided, *dts* must also be provided.

                dts: `List[str]`, optional
                   Datetimes. The corresponding date and time for the temporal
                   dimension of the streamflow observations and predictions.
                   The date and time must be specified in a string following
                   the ISO 8601-1:2019 standard, i.e. "YYYY-MM-DD hh:mm:ss"
                   (e.g. the 21st of May 2007 at 4 in the afternoon is
                   "2007-05-21 16:00:00"). If provided, it is only used if
                   *bootstrap* is also provided.
                   shape: (time,)

            :Returns:

                `List[numpy.ndarray]`
                   The sequence of evaluation metrics computed
                   in the same order as given in *metrics*.
                   shape: [(sites, lead times, subsets, samples, {quantiles,} {thresholds,} {components}), ...]
        )pbdoc",
        py::arg("q_obs"), py::arg("q_prd"), py::arg("metrics"),
        py::arg("q_thr") = xt::pytensor<double, 2>({0}),
        py::arg("t_msk") = xt::pytensor<bool, 4>({0}),
        py::arg("m_cdt") = xt::pytensor<std::array<char, 32>, 2>({0}),
        py::arg("bootstrap") =
            py::dict("n_samples"_a=-9, "len_sample"_a=-9, "summary"_a=0),
        py::arg("dts") = py::list()
    );

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
