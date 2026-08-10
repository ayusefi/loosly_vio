add_library(lvio_core
  common/build_info.cpp
)

target_include_directories(lvio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
)

target_include_directories(lvio_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(lvio_core PUBLIC
  Eigen3::Eigen
  sophus
  Ceres::ceres
  spdlog::spdlog
  ${LVIO_YAML_TARGET}
  ${OpenCV_LIBS}
)

target_compile_definitions(lvio_core PRIVATE LVIO_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

lvio_set_warnings(lvio_core)
