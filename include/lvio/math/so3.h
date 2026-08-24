#pragma once

#include <cmath>

#include "lvio/common/types.h"

namespace lvio {

// Below this threshold, exp and log uses series expansions.
constexpr double kSmallAngle = 1e-2;

// log switches to near 180 degrees branch at this threshold.
constexpr double kNearPi = 1e-8;

// Skew-symmetric matrix of v, built so that Hat(a) * b equals a.cross(b).
inline Mat3 Hat(const Vec3& v) {
  Mat3 m;
  // clang-format off
  m <<    0.0, -v.z(),  v.y(),
       v.z(),    0.0, -v.x(),
      -v.y(),  v.x(),    0.0;
  // clang-format on
  return m;
}

// Reads the three numbers back out of a skew-symmetric matrix. The lower
// triangle is ignored, so a matrix that is not skew-symmetric gives a
// meaningless answer rather than an error.
inline Vec3 Vee(const Mat3& m) { return Vec3(m(2, 1), m(0, 2), m(1, 0)); }

// Rodrigues' formula. Turns a rotatoin vector into a rotation matrix.
inline Mat3 Exp(const Vec3& w) {
  const double theta = w.norm();
  const Mat3 W = Hat(w);

  double a = 0.0;
  double b = 0.0;

  if (theta < kSmallAngle) {
    a = 1.0 - theta * theta / 6.0 + theta * theta * theta * theta / 120.0;
    b = 0.5 - theta * theta / 24.0 + theta * theta * theta * theta / 720.0;
  } else {
    a = std::sin(theta) / theta;
    b = (1.0 - std::cos(theta)) / (theta * theta);
  }

  return Mat3::Identity() + a * W + b * W * W;
}

// Recovers the rotation vector from a roation matrix. Inverse of Exp.
inline Vec3 Log(const Mat3& R) {
  // It already points along the axis. It is equal to 2 * sin(theta) * u.
  const Vec3 w2s = Vee(R - R.transpose());

  const double s = std::abs(w2s.norm() / 2.0);
  const double c = (R.trace() - 1.0) / 2.0;
  const double theta = std::atan2(s, c);

  if (theta < kSmallAngle) {
    const double t2 = theta * theta;
    return (w2s / 2.0) * (1.0 + t2 / 6.0 + 7.0 * t2 * t2 / 360.0);
  } else if (theta > M_PI - kNearPi) {
    // At 180 degrees, axis has left the w2s entirely and lives at R + I.
    const Mat3 uut = 0.5 * (R + Mat3::Identity());

    int i = 0;
    if (uut(1, 1) > uut(i, i)) i = 1;
    if (uut(2, 2) > uut(i, i)) i = 2;

    Vec3 u = uut.col(i) / std::sqrt(uut(i, i));
    u.normalize();

    // The sign is undetermined at exactly 180 degrees. just below it,
    // w2s still carries the sign. Just match it and the function will be
    // continuous.
    if (u.dot(w2s) < 0.0) u = -u;

    return theta * u;
  }

  return (theta / (2.0 * s)) * w2s;
}

// Right Jacobian of SO(3). See Barfoot, "State Estimation for Robotics", 2017.
inline Mat3 RightJacobian(const Vec3& theta) {
  const double t = theta.norm();
  const double t2 = t * t;
  const Mat3 W = Hat(theta);

  double a = 0.0;
  double b = 0.0;

  if (t < kSmallAngle) {
    a = -(0.5 - t2 / 24.0);
    b = 1.0 / 6.0 - t2 / 120.0;
  } else {
    a = -(1.0 - std::cos(t)) / t2;
    b = (t - std::sin(t)) / (t2 * t);
  }

  return Mat3::Identity() + a * W + b * W * W;
}

inline Mat3 RightJacobianInverse(const Vec3& theta) {
  const double t = theta.norm();
  const double t2 = t * t;
  const Mat3 W = Hat(theta);

  const double b =
      (t < kSmallAngle)
          ? 1.0 / 12.0 + t2 / 720.0
          : 1.0 / t2 - (1.0 + std::cos(t)) / (2.0 * t * std::sin(t));

  return Mat3::Identity() + 0.5 * W + b * W * W;
}

}  // namespace lvio
