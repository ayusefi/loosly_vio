# loosly_vio

Loosely-coupled stereo visual-inertial odometry.

An error-state Kalman filter fuses IMU propagation with relative pose
measurements produced by a self-contained stereo visual odometry pipeline. The
two halves stay separate: the filter never sees a pixel, and the VO never sees
an IMU sample. Target dataset is EuRoC MAV.

## Build

On Ubuntu 22.04:

    sudo apt install build-essential cmake libeigen3-dev libopencv-dev \
        libceres-dev libyaml-cpp-dev libspdlog-dev libgtest-dev

Sophus is vendored:

    git submodule update --init

    cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
    cmake --build build/release -j
    ctest --test-dir build/release

Build types are `Release`, `Debug`, `RelWithDebInfo` and `ASan`, the last
enabling the address and undefined-behaviour sanitizers.

## Dataset

    ./scripts/fetch_euroc.sh machine_hall vicon_room1

Extracts to `data/euroc/`, which is not tracked. The archives are roughly
19 GB together and are deleted after extraction unless `KEEP_ARCHIVES=1`.
