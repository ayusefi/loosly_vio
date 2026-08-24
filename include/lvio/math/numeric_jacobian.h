#pragma once

#include "lvio/common/types.h"
#include "lvio/math/pose.h"
#include "lvio/math/so3.h"

namespace lvio {

// Jacobian of f with respect to the body-frame applied on the right of R,
// computed by central difference. It is slow and approximate, but it requires
// no algebra, so it cannot repeat algebric mistakes made in analytic form.
template <typename Function>
inline Mat3 NumericJacobianSo3(const Function& f, const Mat3& R,
                               double h = 1e-6) {
  Mat3 Jacobian;

  for (int i = 0; i < 3; ++i) {
    Vec3 step = Vec3::Zero();
    step(i) = h;

    const Vec3 plus = f(R * Exp(step));
    const Vec3 minus = f(R * Exp(-step));
    Jacobian.col(i) = (plus - minus) / (2.0 * h);
  }
  return Jacobian;
}

// Jacobian of f with respect to a six-number pose nudge, computed by central
// differences. See Perturb for the convention.
template <typename Function>
inline Mat36 NumericJacobianPose(const Function& f, const Pose& pose,
                                 double h = 1e-6) {
  Mat36 jacobian;

  for (int i = 0; i < 6; ++i) {
    Vec6 step = Vec6::Zero();
    step(i) = h;

    const Vec3 plus = f(Perturb(pose, step));
    const Vec3 minus = f(Perturb(pose, -step));

    jacobian.col(i) = (plus - minus) / (2.0 * h);
  }

  return jacobian;
}

}  // namespace lvio