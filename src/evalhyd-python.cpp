#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "evalhyd/nse.hpp"

namespace py = pybind11;

// Python Module and Docstrings

PYBIND11_MODULE(evalhyd, m)
{
    xt::import_numpy();

    m.doc() = "evaluator for hydrological simulations/forecasts";

    m.def("nse", eh::nse<xt::pyarray<double>>, "Return the Nash-Sutcliffe Efficiency (NSE)",
          py::arg("sim"), py::arg("obs"), py::arg("axis") = 0);
}
