#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "evalhyd/nse.hpp"

namespace py = pybind11;

// Python Module and Docstrings

PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = "evaluator for hydrological simulations/forecasts";

    m.def("nse", evalhyd::nse<xt::pytensor<double, 1>>, "Return the Nash-Sutcliffe Efficiency (NSE) [1D arrays]",
          py::arg("sim"), py::arg("obs"), py::arg("axis") = 0);
    m.def("nse", evalhyd::nse<xt::pytensor<double, 2>>, "Return the Nash-Sutcliffe Efficiency (NSE) [2D arrays]",
          py::arg("sim"), py::arg("obs"), py::arg("axis") = 0);
}
