cmake_minimum_required(VERSION 3.15)
project(rms_cpp)

# ── C++ standard ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release flags ─────────────────────────────────────────────
# -O3       : full optimisation
# -march=native : use all CPU features of the build machine
# -ffast-math   : allow float reassociation (fine for MC work)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native)

# ── Find Python + pybind11 ────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)

# ── Source files ──────────────────────────────────────────────
set(SOURCES
    src/indicators.cpp
    src/samplers.cpp
    src/forecast.cpp
    src/bindings.cpp
)

# ── Build the shared library ──────────────────────────────────
pybind11_add_module(rms_cpp ${SOURCES})

target_include_directories(rms_cpp PRIVATE
    ${Python3_NumPy_INCLUDE_DIRS}
    include
)

# ── Output the .so next to the Python source so it can be
#    imported directly without install ─────────────────────────
set_target_properties(rms_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)
