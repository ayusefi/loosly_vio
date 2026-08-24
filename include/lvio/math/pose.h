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

}  // namespace lvio
