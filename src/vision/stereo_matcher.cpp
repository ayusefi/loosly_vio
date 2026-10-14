#include "lvio/vision/stereo_matcher.h"

#include "lvio/math/so3.h"

#include <cmath>

namespace lvio {

    std::optional<Vec3> TriangulateMidpoint(const Vec3& left_ray,
                                            const Vec3& right_ray,
                                            const Vec3& baseline) {
        const double aa = left_ray.dot(left_ray);
        const double ab = left_ray.dot(right_ray);
        const double bb = right_ray.dot(right_ray);

        // Cauchy-Schwarz guarantees this is never negative, and for unit rays it is
        // exactly sin^2 of the parallax angle. If it ever goes negative the
        // subtraction is the wrong way round. See lesson 49 part 3 step 10.
        const double det = aa * bb - ab * ab;
        // Parallel rays: sin(parallax) below 1e-6. An infinitely distant point
        // produces no jump at all, so there is nothing to compute.
        if (det < 1e-12) {
            return std::nullopt;
        }

        const double at = left_ray.dot(baseline);
        const double bt = right_ray.dot(baseline);

        const double s = (at * bb - ab * bt) / det;
        const double u = (ab * at - aa * bt) / det;
        
        // The closest point on each ray, averaged. They do not meet, because the
        // measurement is noisy; on real data they pass within about 1.5 mm.
        const Vec3 on_left = s * left_ray;
        const Vec3 on_right = baseline + u * right_ray;
        return 0.5 * (on_left + on_right);
    }

    std::vector<StereoPoint> MatchStereo(const cv::Mat& left, const cv::Mat& right,
                                         const std::vector<FeatureTrack>& tracks,
                                         const StereoCalibration& calibration, 
                                         const StereoMatcherConfig& config) {
        std::vector<StereoPoint> points;
        if(tracks.empty()) {
            return points;
        }

        std::vector<cv::Point2f> pixels;
        pixels.reserve(tracks.size());
        for (const FeatureTrack& t : tracks) {
            pixels.push_back(t.point);
        }

        // The KLT. Two cameras at one instant rather than one camera at two instants; 
        // the method does not care.
        const TrackResult matched = TrackFeatures(left, right, pixels, config.tracker);

        const Mat3 R = calibration.T_left_right.R;
        const Vec3 t = calibration.T_left_right.t;
        // x_left^T E x_right = 0. Getting this order backwards gives a residual with
        // a median of 10 px and a spread of 1 px -- a bias, not noise.
        const Mat3 E = Hat(t) * R;

        points.reserve(tracks.size());
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (!matched.ok[i]) {
                 continue;
            }

            const cv::Point2f& rp = matched.points[i];

            // Rays, with intrinsics and lens distortion both removed. Our Unproject
            // round trips to 2e-06 px where cv::undistortPoints manages only 0.29 px.
            const Vec3 ray_left = calibration.left.Unproject(Vec2(tracks[i].point.x, tracks[i].point.y));
            const Vec3 ray_right_own = calibration.right.Unproject(Vec2(rp.x, rp.y));

            // Into left axes. The two cameras are 0.8184 degrees apart, which is up to
            // 6.5 px of image shift, so this rotation is not optional.
            const Vec3 ray_right = R * ray_right_own;

            // Gate 2. E turns the left ray into the epipolar line in the right image,
            // as (A, B, C) meaning A*x + B*y + C = 0. Then the schoolbook point-to-line
            // distance, scaled by the focal length to get pixels. Pushing a correct
            // match 1 px off its line makes this read 1.0000, so the threshold means
            // exactly what it says.
            const Vec3 na = ray_left / ray_left.z();
            const Vec3 nb = ray_right_own / ray_right_own.z();
            const Vec3 line = E.transpose() * na;
            const double epipolar = std::abs(line.dot(nb)) /
                                    std::sqrt(line.x() * line.x() + line.y() * line.y()) *
                                    calibration.left.fu;
            if (epipolar > config.max_epipolar) {
                continue;
            }
            const std::optional<Vec3> point = TriangulateMidpoint(ray_left, ray_right, t);
            if (!point.has_value()) {
                continue;
            }

            // Gate 1 and gate 4, together.
            if (point->z() < config.min_depth || point->z() > config.max_depth) {
                continue;
            }

            // Gate 3: rebuild the pixel from the point we just made and compare.
            const double reprojection = (calibration.left.Project(*point) - Vec2(tracks[i].point.x, tracks[i].point.y)).norm();
            if (reprojection > config.max_reprojection) {
                continue;
            }

            points.push_back({tracks[i].id, tracks[i].point, rp, *point});
        }

        return points;

    }
} // namespace lvio