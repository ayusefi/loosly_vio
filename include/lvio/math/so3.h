#pragma once

#include "lvio/common/types.h"
#include <cmath>

namespace lvio {

// Below this threshold, exp and log uses series expansions.
constexpr double kSmallAngle = 1e-2;

// Skew-symmetric matrix of v, built so that Hat(a) * b equals a.cross(b).
inline Mat3 Hat(const Vec3& v) {
  Mat3 m;
  m <<    0.0, -v.z(),  v.y(),
       v.z(),    0.0, -v.x(),
      -v.y(),  v.x(),    0.0;
  return m;
}

// Reads the three numbers back out of a skew-symmetric matrix. The lower
// triangle is ignored, so a matrix that is not skew-symmetric gives a
// meaningless answer rather than an error.
inline Vec3 Vee(const Mat3& m) {
  return Vec3(m(2, 1), m(0, 2), m(1, 0));
}

// Rodrigues' formula. Turns a rotatoin vector into a rotation matrix.
inline Mat3 Exp(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 W = Hat(w);

  double a = 0.0;
  double b = 0.0;

  if (theta < kSmallAngle) {
    a = 1.0 - theta * theta / 6.0 + theta * theta * theta * theta / 120.0;
    b = 0.5 - theta * theta / 24.0 + theta * theta * theta * theta / 720.0;
  }
  else {
    a = std::sin(theta) / theta;
    b = (1.0 - std::cos(theta)) / (theta * theta);
  }

  return Mat3::Identity() + a * W + b * W * W;

}

}  // namespace lvio
