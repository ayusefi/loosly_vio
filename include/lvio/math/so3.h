#pragma once

#include "lvio/common/types.h"

namespace lvio {

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

}  // namespace lvio
