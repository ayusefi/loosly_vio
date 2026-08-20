#pragma once

#include "lvio/common/types.h"
#include "lvio/math/so3.h"

namespace lvio {

    // Jacobian of f with respect to the body-frame applied on the right of R, computed by central difference.
    // It is slow and approximate, but it requires no algebra, so it cannot repeat algebric mistakes made in analytic form.
    template <typename Function>
    inline Mat3 NumericJacobianSo3(const Function& f, const Mat3& R, 
                                   double h=1e-6) {
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

} // namespace lvio