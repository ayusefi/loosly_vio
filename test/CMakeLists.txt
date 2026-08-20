function(lvio_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE lvio_core GTest::gtest GTest::gtest_main)
  lvio_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

lvio_add_test(unit_toolchain)
lvio_add_test(unit_so3)
lvio_add_test(unit_numeric_jacobian)

