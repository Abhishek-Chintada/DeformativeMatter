cmake_minimum_required(VERSION 3.10)
project(fem_sheet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(fem_sheet fem_sheet.cpp)
target_link_libraries(fem_sheet PRIVATE Eigen3::Eigen glfw OpenGL::GL)
if(APPLE)
  target_compile_definitions(fem_sheet PRIVATE GL_SILENCE_DEPRECATION)
endif()
