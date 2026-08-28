#include "lvio/camera/camera.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <vector>

namespace lvio {
namespace {
    std::vector<double> RequireList(const YAML::Node& node, const std::string& key, 
                                     std::size_t expected, const std::string& path) {
        if (!node[key]) {
            throw std::runtime_error("Missing " + key + " in " + path);
        }
        const auto values = node[key].as<std::vector<double>>();
        if (values.size() != expected) {
            throw std::runtime_error("'" + key + "' has wrong length in " + path);
        }

        return values;
    }
} // namespace

Vec2 Camera::Distort(const Vec2& point) const {

    const double x = point.x();
    const double y = point.y();

    const double r2 = x * x + y * y; // r2 = x2 + y2
    const double scale = 1 + k1 * r2 + k2 * r2 * r2; //   scale  =  1  +  k1 * r²  +  k2 * r⁴

    const double twist_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double twist_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    return Vec2(x * scale + twist_x, y * scale + twist_y);
}

Vec2 Camera::Project(const Vec3& point) const {
    
    if (point.z() <= 0.0) {
        throw std::runtime_error("cannot project a point at or behind the camera");
    }

    const Vec2 on_screen = Vec2(point.x()/point.z(), point.y()/point.z());
    const Vec2 bent = Distort(on_screen);

    return Vec2(fu * bent.x() + cu, fv * bent.y() + cv);
}

Camera ReadCamera(const std::string& path) {
    const YAML::Node node = YAML::LoadFile(path);

    const auto model = node["camera_model"].as<std::string>();
    if (model != "pinhole") {
        throw std::runtime_error("unsupported camera_model '" + model + "' in " +
                                path);
    }
    const auto distortion_model = node["distortion_model"].as<std::string>();
    if (distortion_model != "radial-tangential") {
        throw std::runtime_error("unsupported distortion_model '" +
                                distortion_model + "' in " + path);
    }

    const std::vector<double> resolution = RequireList(node, "resolution", 2, path);
    const std::vector<double> intrinsics = RequireList(node, "intrinsics", 4, path);
    const std::vector<double> distortion =
        RequireList(node, "distortion_coefficients", 4, path);

    Camera camera;
    camera.width = static_cast<int>(resolution[0]);
    camera.height = static_cast<int>(resolution[1]);
    camera.fu = intrinsics[0];
    camera.fv = intrinsics[1];
    camera.cu = intrinsics[2];
    camera.cv = intrinsics[3];
    camera.k1 = distortion[0];
    camera.k2 = distortion[1];
    camera.p1 = distortion[2];
    camera.p2 = distortion[3];

    return camera;
}


} // namespace lvio