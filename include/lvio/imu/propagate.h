#pragma once

#include "lvio/common/time.h"
#include "lvio/common/types.h"
#include "lvio/io/euroc.h"
#include "lvio/math/pose.h"

namespace lvio {
// Gravity in the world frame, pointing down.
inline const Vec3 kGravity(0.0, 0.0, -9.81);

// Orientation, velocity and position at one instant. The pose is T_WB and the
// velocity is in world axes.
struct NavState {
  Timestamp time = 0;
  Pose pose;
  Vec3 velocity = Vec3::Zero();
};

// The two constant offsets the sensor adds to its readings.
struct ImuBias {
  Vec3 accel = Vec3::Zero();
  Vec3 gyro = Vec3::Zero();
};

// Advances the state by one IMU sample, using the reading at the start of the
// step and holding it for the whole step.
NavState Propagate(const NavState& state, const ImuSample& sample,
                   const ImuBias& bias, double dt);

// The four noise numbers from imu0/sensor.yaml. Densities, not standard
// deviations -- see lesson 37 part 5 for the conversion.
struct ImuNoise {
  double accel_density = 2.0000e-3;      // m/s^2 / sqrt(Hz)
  double gyro_density = 1.6968e-4;       // rad/s / sqrt(Hz)
  double accel_random_walk = 3.0000e-3;  // m/s^3 / sqrt(Hz)
  double gyro_random_walk = 1.9393e-5;   // rad/s^2 / sqrt(Hz)
};

// How the fifteen errors turn into each other over one step.
Mat15 ErrorTransition(const NavState& state, const ImuSample& sample,
                      const ImuBias& bias, double dt);

// The fresh error one step adds. Diagonal, because the four noise
// sources are assumed unrelated.
Mat15 ProcessNoise(const ImuNoise& noise, double dt);

}  // namespace lvio