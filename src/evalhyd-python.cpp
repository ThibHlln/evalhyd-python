#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define FORCE_IMPORT_ARRAY
#include <xtensor/xexpression.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/evald.hpp"
#include "evalhyd/evalp.hpp"

namespace py = pybind11;
using namespace py::literals;

// reshape 1D tensors to 2D tensors
auto evald_1d(
    const xt::pytensor<double, 1>& q_obs,
    const xt::pytensor<double, 1>& q_prd,
    const std::vector<std::string>& metrics,
    const std::string& transform,
    const double exponent,
    double epsilon,
    const xt::pytensor<bool, 2>& t_msk,
    const xt::pytensor<std::array<char, 32>, 1>& m_cdt,
    const std::unordered_map<std::string, int>& bootstrap,
    const std::vector<std::string>& dts
)
{
    return evalhyd::evald<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>(
        xt::pytensor<double, 2>(xt::view(q_obs, xt::newaxis(), xt::all())),
        xt::pytensor<double, 2>(xt::view(q_prd, xt::newaxis(), xt::all())),
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

    m.doc() = "Utility for evaluation of streamflow predictions";

    // deterministic evaluation
    m.def(
        "evald",
        &evald_1d,
        "Function to evaluate deterministic streamflow predictions (1D)",
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
        "evald",
        &evalhyd::evald<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
        "Function to evaluate deterministic streamflow predictions (2D)",
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
        "evalp",
        &evalhyd::evalp<xt::pytensor<double, 2>, xt::pytensor<double, 4>, xt::pytensor<bool, 4>>,
        "Function to evaluate probabilistic streamflow predictions",
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
