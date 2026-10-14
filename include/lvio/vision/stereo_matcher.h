#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

#include "lvio/camera/camera.h"
#include "lvio/common/types.h"
#include "lvio/vision/feature_tracker.h"

namespace lvio {

// The gates: max_depth is the depth at which this baseline can no longer 
// measure to 10%, which works out at 17 m -- very close to the 16.56 m 99th
// percentile of what the camera actually sees here. max_epipolar is in 
// pixels off the epipolar line, eight times the measured median of 0.25 px, 
// so it removes the tail and not the core.
struct StereoMatcherConfig {
    TrackerConfig tracker;
    double min_depth = 0.1;
    double max_depth = 17.0;
    double max_epipolar = 2.0;
    double max_reprojection = 1.5;
};

// A left feature matched into the right image and turned into a 3D point. The
// id is the left track's id, so a point keeps its name across frames.
struct StereoPoint {
    int id = 0;
    cv::Point2f left;
    cv::Point2f right;
    Vec3 point = Vec3::Zero(); // in left camera axis, metres
};

// Where two rays come closest, averaged. `left_ray` starts at the origin and
// `right_ray` at `baseline`, both expressed in left camera axes. Returns
// nothing when the rays are parallel, which is what an infinitely far point
// looks like -- that is physics, not a defensive check.
std::optional<Vec3> TriangulateMidpoint(const Vec3& left_ray, 
                                        const Vec3& right_ray, 
                                        const Vec3& baseline);

// Matches each track into the right image and triangulates it. Only points
// passing every gate are returned, so the result is shorter than `tracks`.
std::vector<StereoPoint> MatchStereo(const cv::Mat& left, const cv::Mat& right,
                                     const std::vector<FeatureTrack>& tracks,
                                     const StereoCalibration& calibration,
                                     const StereoMatcherConfig& config);

} // namespace lvio