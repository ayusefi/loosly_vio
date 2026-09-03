#include "lvio/imu/propagate.h"

#include "lvio/math/so3.h"

namespace lvio {
NavState Propagate(const NavState& state, const ImuSample& sample,
                   const ImuBias& bias, double dt) {
  // Subtract the bias, rotate into world axes, add back the
  // gravity the sensor could not feel.
  const Vec3 turn = sample.gyro - bias.gyro;
  ;
  const Vec3 force = sample.accel - bias.accel;
  const Vec3 acceleration = state.pose.R * force + kGravity;

  NavState next;
  next.time = state.time + FromSeconds(dt);

  // Position uses the OLD velocity, so update it before overwriting v.
  next.pose.t =
      state.pose.t + state.velocity * dt + 0.5 * acceleration * dt * dt;
  next.velocity = state.velocity + acceleration * dt;
  next.pose.R = state.pose.R * Exp(turn * dt);

  return next;
}
}  // namespace lvio