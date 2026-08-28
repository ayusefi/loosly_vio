#pragma once

#include <string>

#include "lvio/common/types.h"

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
    };

    // Reads one of the mav0/camN/sensor.yaml files. Throws if a key is missing or if
    // the file describes a camera or distortion model this code does not implement.
    Camera ReadCamera(const std::string& path);

} // namespace lvio