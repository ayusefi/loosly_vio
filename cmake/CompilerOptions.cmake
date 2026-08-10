# ASan/UBSan build type. -O1 keeps it usable on full sequences; the sanitizers
# are useless at -O0 speed once we start replaying 200 Hz IMU data.
set(CMAKE_CXX_FLAGS_ASAN
  "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all"
  CACHE STRING "Flags used by the C++ compiler for ASan builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN
  "-fsanitize=address,undefined"
  CACHE STRING "Flags used by the linker for ASan builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN
  "-fsanitize=address,undefined"
  CACHE STRING "Flags used by the linker for ASan builds" FORCE)
mark_as_advanced(CMAKE_CXX_FLAGS_ASAN CMAKE_EXE_LINKER_FLAGS_ASAN
                 CMAKE_SHARED_LINKER_FLAGS_ASAN)

option(LVIO_NATIVE_ARCH "Build for the host CPU" ON)
if(LVIO_NATIVE_ARCH)
  add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

# Applied to our own targets only. Dependency headers come in as SYSTEM so
# their warnings do not fail the build.
function(lvio_set_warnings target)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Woverloaded-virtual
    -Wdouble-promotion
    -Wnull-dereference)
endfunction()
