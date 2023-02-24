#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <optional>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define FORCE_IMPORT_ARRAY
#include <xtl/xoptional.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/evald.hpp"
#include "evalhyd/evalp.hpp"

namespace py = pybind11;
using namespace py::literals;

auto evald(
    const xt::pytensor<double, 2>& q_obs,
    const xt::pytensor<double, 2>& q_prd,
    const std::vector<std::string>& metrics,
    const xt::pytensor<double, 2>& q_thr,
    std::optional<std::string> events,
    std::optional<std::string> transform,
    std::optional<double> exponent,
    std::optional<double> epsilon,
    const xt::pytensor<bool, 3>& t_msk,
    const xt::pytensor<std::array<char, 32>, 2>& m_cdt,
    std::optional<std::unordered_map<std::string, int>> bootstrap,
    const std::vector<std::string>& dts,
    std::optional<int> seed,
    std::optional<std::vector<std::string>> diagnostics
)
{
    return evalhyd::evald(
        q_obs,
        q_prd,
        metrics,
        q_thr,
        (events.has_value()) ? events.value() : xtl::missing<std::string>(),
        (transform.has_value()) ? transform.value() : xtl::missing<std::string>(),
        (exponent.has_value()) ? exponent.value() : xtl::missing<double>(),
        (epsilon.has_value()) ? epsilon.value() : xtl::missing<double>(),
        t_msk,
        m_cdt,
        (bootstrap.has_value())
        ? bootstrap.value()
        : xtl::missing<std::unordered_map<std::string, int>>(),
        dts,
        (seed.has_value()) ? seed.value() : xtl::missing<int>(),
        (diagnostics.has_value())
        ? diagnostics.value()
        : xtl::missing<std::vector<std::string>>()
    );
}

auto evalp(
    const xt::pytensor<double, 2>& q_obs,
    const xt::pytensor<double, 4>& q_prd,
    const std::vector<std::string>& metrics,
    const xt::pytensor<double, 2>& q_thr,
    std::optional<std::string> events,
    const std::vector<double>& c_lvl,
    const xt::pytensor<bool, 4>& t_msk,
    const xt::pytensor<std::array<char, 32>, 2>& m_cdt,
    std::optional<std::unordered_map<std::string, int>> bootstrap,
    const std::vector<std::string>& dts,
    std::optional<int> seed,
    std::optional<std::vector<std::string>> diagnostics
)
{
    return evalhyd::evalp(
        q_obs,
        q_prd,
        metrics,
        q_thr,
        (events.has_value()) ? events.value() : xtl::missing<std::string>(),
        c_lvl,
        t_msk,
        m_cdt,
        (bootstrap.has_value())
        ? bootstrap.value()
        : xtl::missing<std::unordered_map<std::string, int>>(),
        dts,
        (seed.has_value()) ? seed.value() : xtl::missing<int>(),
        (diagnostics.has_value())
        ? diagnostics.value()
        : xtl::missing<std::vector<std::string>>()
    );
}

// Python Module and Docstrings
PYBIND11_MODULE(_evalhyd, m)
{
    xt::import_numpy();

    m.doc() = "Python bindings for the C++ core of evalhyd";

    // deterministic evaluation
    m.def(
        "_evald",
        &evald,
        "Function to evaluate determinist streamflow predictions (2D)",
        py::arg("q_obs"),
        py::arg("q_prd"),
        py::arg("metrics"),
        py::arg("q_thr") = xt::pytensor<double, 2>({0}),
        py::arg("events") = py::none(),
        py::arg("transform") = py::none(),
        py::arg("exponent") = py::none(),
        py::arg("epsilon") = py::none(),
        py::arg("t_msk") = xt::pytensor<bool, 3>({0}),
        py::arg("m_cdt") = xt::pytensor<std::array<char, 32>, 2>({0}),
        py::arg("bootstrap") = py::none(),
        py::arg("dts") = py::list(),
        py::arg("seed") = py::none(),
        py::arg("diagnostics") = py::none()
    );

    // probabilistic evaluation
    m.def(
        "_evalp",
        &evalp,
        "Function to evaluate probabilist streamflow predictions",
        py::arg("q_obs"),
        py::arg("q_prd"),
        py::arg("metrics"),
        py::arg("q_thr") = xt::pytensor<double, 2>({0}),
        py::arg("events") = py::none(),
        py::arg("c_lvl") = py::list(),
        py::arg("t_msk") = xt::pytensor<bool, 4>({0}),
        py::arg("m_cdt") = xt::pytensor<std::array<char, 32>, 2>({0}),
        py::arg("bootstrap") = py::none(),
        py::arg("dts") = py::list(),
        py::arg("seed") = py::none(),
        py::arg("diagnostics") = py::none()
    );

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
