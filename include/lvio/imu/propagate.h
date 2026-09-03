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

}  // namespace lvio