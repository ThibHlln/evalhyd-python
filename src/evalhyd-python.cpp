#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/determinist.hpp"
#include "evalhyd/probabilist.hpp"

namespace py = pybind11;

namespace ehd = evalhyd::determinist;
namespace ehp = evalhyd::probabilist;

// Python Module and Docstrings
PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = "evaluator for hydrological simulations/forecasts";

    // Submodule for deterministic evaluation of streamflow simulations
    py::module_ md = m.def_submodule("determinist", "deterministic streamflow evaluation");

    md.def(
        "evaluate", ehd::evaluate<xt::pytensor<double, 1>>,
        "Deterministic streamflow evaluation [1D arrays]",
        py::arg("metrics"), py::arg("q_obs"), py::arg("q_sim")
    );
    md.def(
        "evaluate", ehd::evaluate<xt::pytensor<double, 2>>,
        "Deterministic streamflow evaluation [2D arrays]",
        py::arg("metrics"), py::arg("q_obs"), py::arg("q_sim")
    );

    // Submodule for probabilistic evaluation of streamflow forecasts
    py::module_ mp = m.def_submodule("probabilist", "probabilistic streamflow evaluation");

    mp.def(
        "evaluate", ehp::evaluate,
        "Probabilist streamflow evaluation",
        py::arg("metrics"), py::arg("q_obs"), py::arg("q_frc"), py::arg("q_thr")
    );
}
