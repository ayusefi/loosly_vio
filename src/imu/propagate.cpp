#include "lvio/imu/propagate.h"

#include "lvio/math/so3.h"

namespace lvio {

namespace {
constexpr int kIdxP = 0;
constexpr int kIdxV = 3;
constexpr int kIdxTheta = 6;
constexpr int kIdxBa = 9;
constexpr int kIdxBg = 12;
}  // namespace

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

Mat15 ErrorTransition(const NavState& state, const ImuSample& sample,
                      const ImuBias& bias, double dt) {
  const Vec3 force = sample.accel - bias.accel;
  const Vec3 turn = sample.gyro - bias.gyro;
  const Mat3& R = state.pose.R;

  Mat15 a = Mat15::Identity();

  // A velocity error becomes a position error.
  a.block<3, 3>(kIdxP, kIdxV) = Mat3::Identity() * dt;

  // The orientation error rotates but does not grow.
  a.block<3, 3>(kIdxTheta, kIdxTheta) = Exp(-turn * dt);

  // A gyro bias error becomes an orientation error.
  a.block<3, 3>(kIdxTheta, kIdxBg) = -Mat3::Identity() * dt;

  // A tilt misdirects gravity.
  a.block<3, 3>(kIdxV, kIdxTheta) = -R * Hat(force) * dt;

  // The R converts the bias from body axes into world axes.
  a.block<3, 3>(kIdxV, kIdxBa) = -R * dt;

  // The same two effects reach the position through the
  // half-a-dt-squared term. Small, but they improve the test by 4500x.
  a.block<3, 3>(kIdxP, kIdxTheta) = -0.5 * R * Hat(force) * dt * dt;
  a.block<3, 3>(kIdxP, kIdxBa) = -0.5 * R * dt * dt;

  return a;
}

Mat15 ProcessNoise(const ImuNoise& noise, double dt) {
  Mat15 q = Mat15::Zero();

  // Each block is (density)^2 * dt.
  q.block<3, 3>(kIdxV, kIdxV) =
      Mat3::Identity() * noise.accel_density * noise.accel_density * dt;
  q.block<3, 3>(kIdxTheta, kIdxTheta) =
      Mat3::Identity() * noise.gyro_density * noise.gyro_density * dt;
  q.block<3, 3>(kIdxBa, kIdxBa) =
      Mat3::Identity() * noise.accel_random_walk * noise.accel_random_walk * dt;
  q.block<3, 3>(kIdxBg, kIdxBg) =
      Mat3::Identity() * noise.gyro_random_walk * noise.gyro_random_walk * dt;

  return q;
}
}  // namespace lvio