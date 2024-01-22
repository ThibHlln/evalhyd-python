cmake_minimum_required(VERSION 3.15)

project(
        EvalHyd-Python
        LANGUAGES CXX C
        VERSION 0.1.2.0
        DESCRIPTION "Python bindings for evalhyd utility"
)

add_library(
        evalhyd-python MODULE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/evalhyd-python.cpp
)

# ------------------------------------------------------------------------------
# dependencies and build
# ------------------------------------------------------------------------------

if(SKBUILD)
        find_package(PythonExtensions REQUIRED)
        find_package(NumPy REQUIRED)
        python_extension_module(evalhyd-python)
else()
        find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
        target_link_libraries(evalhyd-python Python::NumPy)
        # use only header if numpy target links to libpython
        #target_include_directories(evalhyd-python SYSTEM PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endif()

find_package(xtensor 0.7.5 REQUIRED)
message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")

find_package(xtensor 0.24.7 REQUIRED)
message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")

find_package(xtensor-python 0.26.1 REQUIRED)
message(STATUS "Found xtensor-python: ${xtensor-python_INCLUDE_DIRS}/xtensor-python")

find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}/pybind11")

if(DEFINED EVALHYD_SRC)
        set(EVALHYD_BUILD_TEST OFF CACHE BOOL "configure and compile tests")
        add_subdirectory(${EVALHYD_SRC} deps/evalhyd)
else()
        find_package(EvalHyd 0.1.2 REQUIRED)
endif()

target_link_libraries(
        evalhyd-python
        EvalHyd::evalhyd
        xtensor-python
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
)

set_target_properties(
        evalhyd-python PROPERTIES
        OUTPUT_NAME evalhyd
)

pybind11_extension(evalhyd-python)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
        # Strip unnecessary sections of the binary on Linux/macOS
        pybind11_strip(evalhyd-python)
endif()

set_target_properties(
        evalhyd-python PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
)

# add include directories
target_include_directories(
        evalhyd-python
        PUBLIC
                ${NUMPY_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# installation
# ------------------------------------------------------------------------------

if(SKBUILD)
        install(
                TARGETS evalhyd-python
                DESTINATION evalhyd-python
        )
endif()
