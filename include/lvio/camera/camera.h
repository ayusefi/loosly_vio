#pragma once

#include <string>

#include "lvio/common/types.h"
#include "lvio/math/pose.h"

namespace lvio {

// A pinhole camera with radial-tangential distortion, which is what EuRoC uses.
// The camera looks along its own +z, u grows right and v grows down.
struct Camera {
  int height = 0;
  int width = 0;

  double fu = 0.0;
  double fv = 0.0;
  double cu = 0.0;
  double cv = 0.0;

  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;

  // Applies the lens distortion to a point on the unit screen.
  Vec2 Distort(const Vec2& point) const;

  // A 3D point in camera coordinates to a pixel. Divide by depth, distort,
  // scale to pixels. Requires point.z() > 0: a point level with or behind the
  // camera has no image.
  Vec2 Project(const Vec3& point) const;

  // Undoes Distort. There is no formula, so this guesses and improves;
  // Throws if it has not settled after kMaxRounds, which means the pixel
  // is far outside the image or the coefficients are not sane.
  Vec2 Undistort(const Vec2& distorted) const;

  // A pixel to a unit-length ray in camera coordinates. The depth is gone, so
  // this gives a direction, not a point.
  Vec3 Unproject(const Vec2& pixel) const;
};

// Reads one of the mav0/camN/sensor.yaml files. Throws if a key is missing or
// if the file describes a camera or distortion model this code does not
// implement.
Camera ReadCamera(const std::string& path);

// The two cameras and how they are placed. T_body_left and T_body_right come
// straight from the two sensor.yaml files; T_left_right is derived from them,
// because neither file mentions the other camera.
struct StereoCalibration {
  Camera left;
  Camera right;
  Pose T_body_left;
  Pose T_body_right;
  Pose T_left_right;
};

// Reads the T_BS block out of a sensor.yaml.
Pose ReadExtrinsics(const std::string& path);

// Reads both cameras from a mav0 directory and works out where they sit
// relative to each other.
StereoCalibration ReadStereoCalibration(const std::string& mav0_dir);

}  // namespace lvio