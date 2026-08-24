#pragma once

#include "lvio/common/types.h"
#include "lvio/math/so3.h"

namespace lvio {

    // A rigid transform, rotate and then translate.
    struct Pose {

        Mat3 R = Mat3::Identity();
        Vec3 t = Vec3::Zero();

        Pose() = default;

        Pose(const Mat3& rotation, const Vec3& translation)
            : R(rotation), t(translation) {}

        // P_A = R * P_B + t
        Vec3 operator*(const Vec3& p) const {
            return R * p + t;
        }

        // Composition.
        // Applying the result is same as applying the other, then this.
        Pose operator*(const Pose& pose) const {
            return Pose(
                R * pose.R,
                R * pose.t + t
            );
        }

        // Inverse transform.
        Pose Inverse() const {
            const Mat3 Rt = R.transpose();

            return Pose(
                Rt,
                -Rt * t
            );
        }
    };

    // Transform from pose a to pose b.
    inline Pose Between(const Pose& a, const Pose& b) {
        return a.Inverse() * b;
    }

}