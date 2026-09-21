cmake_minimum_required(VERSION 3.16)
project(FEM_Solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

add_executable(fem_main
    cpp_source_eigen/main.cpp
)

target_link_libraries(fem_main PRIVATE Eigen3::Eigen)