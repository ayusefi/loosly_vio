#pragma once

#include "lvio/common/types.h"
#include "lvio/math/so3.h"

namespace lvio {

// A rigid transform, rotate and then translate. A Pose named T_AB maps points
// from frame B into frame A.
struct Pose {
  Mat3 R = Mat3::Identity();
  Vec3 t = Vec3::Zero();

  Pose() = default;

  Pose(const Mat3& rotation, const Vec3& translation)
      : R(rotation), t(translation) {}

  // p_A = R p_B + t
  Vec3 operator*(const Vec3& p) const { return R * p + t; }

  // Applying the result is the same as applying pose, then this.
  Pose operator*(const Pose& pose) const {
    return Pose(R * pose.R, R * pose.t + t);
  }

  Pose Inverse() const {
    const Mat3 Rt = R.transpose();
    return Pose(Rt, -Rt * t);
  }
};

// Transform from pose a to pose b.
inline Pose Between(const Pose& a, const Pose& b) { return a.Inverse() * b; }

// Apply a six-number nudge to a pose.
inline Pose Perturb(const Pose& pose, const Vec6& delta) {
  return Pose(pose.R * Exp(delta.tail<3>()), pose.t + delta.head<3>());
}

// The pose a fraction s of the way from a to b. s = 0 gives a, s = 1 gives b.
// The position interpolates linearly. The rotation cannot -- averaging rotations
// does not produce a rotation -- so it goes out to the tangent space and back.
inline Pose Interpolate(const Pose& a, const Pose& b, double s) {
  return Pose(a.R * Exp(Log(a.R.transpose() * b.R) * s), a.t + (b.t - a.t) * s);
}

}  // namespace lvio
