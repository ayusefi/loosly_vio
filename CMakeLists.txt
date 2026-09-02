cmake_minimum_required(VERSION 3.16)
project(lvio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo ASan)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CompilerOptions)

option(LVIO_BUILD_TESTS "Build unit tests" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED
  COMPONENTS core imgproc imgcodecs features2d video calib3d highgui)
find_package(Ceres 2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)

# yaml-cpp 0.7 (Ubuntu 22.04) exports a bare target, 0.8 a namespaced one.
if(TARGET yaml-cpp::yaml-cpp)
  set(LVIO_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(LVIO_YAML_TARGET yaml-cpp)
endif()

# Sophus ships a CMake project that pulls in fmt, its own tests and an install
# rule. It is header-only, so consume the headers directly instead. Basic
# logging keeps the fmt dependency out.
add_library(sophus INTERFACE)
target_include_directories(sophus SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Sophus)
target_compile_definitions(sophus INTERFACE SOPHUS_USE_BASIC_LOGGING)
target_link_libraries(sophus INTERFACE Eigen3::Eigen)

configure_file(cmake/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/lvio/common/version.h @ONLY)

add_subdirectory(src)
add_subdirectory(apps)

if(LVIO_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(test)
endif()

message(STATUS "lvio ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  tests=${LVIO_BUILD_TESTS}")
message(STATUS "  Eigen   ${Eigen3_VERSION}")
message(STATUS "  OpenCV  ${OpenCV_VERSION}")
message(STATUS "  Ceres   ${Ceres_VERSION}")
