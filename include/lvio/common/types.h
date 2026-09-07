#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lvio {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat36 = Eigen::Matrix<double, 3, 6>;
using Vec2 = Eigen::Vector2d;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Mat15 = Eigen::Matrix<double, 15, 15>;

}  // namespace lvio
