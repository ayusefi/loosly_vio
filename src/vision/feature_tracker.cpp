#include "lvio/vision/feature_tracker.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include <opencv2/video/tracking.hpp>

#include <cmath>

namespace lvio {

namespace {

// Which cell a pixel falls in. The min() guards the right and bottom edges: a
// point at exactly x == width would otherwise index one cell past the end.
int CellIndex(const cv::Point2f& p, int width, int height,
              const DetectorConfig& config) {
  const int col = std::min(static_cast<int>(p.x * config.grid_cols / width),
                           config.grid_cols - 1);
  const int row = std::min(static_cast<int>(p.y * config.grid_rows / height),
                           config.grid_rows - 1);
  return row * config.grid_cols + col;
}

}  // namespace

std::vector<int> CellCounts(const std::vector<cv::Point2f>& features, int width,
                            int height, const DetectorConfig& config) {
  const std::size_t cells =
      static_cast<std::size_t>(config.grid_cols * config.grid_rows);
  std::vector<int> counts(cells, 0);
  for (const cv::Point2f& p : features) {
    counts[static_cast<std::size_t>(CellIndex(p, width, height, config))] += 1;
  }
  return counts;
}

std::vector<cv::Point2f> DetectFeatures(const cv::Mat& image,
                                        const DetectorConfig& config) {
  // A maxCorners of 0 means "no limit" -- we do our own limiting below.
  // The list comes back sorted, best corner first, which is what lets a single
  // forward walk keep the best few in every cell.
  std::vector<cv::Point2f> pool;
  cv::goodFeaturesToTrack(image, pool, 0, config.quality_level,
                          config.min_distance);

  const std::size_t cells =
      static_cast<std::size_t>(config.grid_cols * config.grid_rows);
  std::vector<int> used(cells, 0);

  std::vector<cv::Point2f> kept;
  kept.reserve(pool.size());
  for (const cv::Point2f& p : pool) {
    const std::size_t cell = static_cast<std::size_t>(
        CellIndex(p, image.cols, image.rows, config));
    if (used[cell] < config.max_per_cell) {
      used[cell]++;
      kept.push_back(p);
    }
  }
  return kept;
}

TrackResult TrackFeatures(const cv::Mat& previous, const cv::Mat& current,
                          const std::vector<cv::Point2f>& features,
                          const TrackerConfig& config) {
  TrackResult result;
  if (features.empty()) {
    return result;  // OpenCV will not accept an empty point list.
  }

  const cv::Size window(config.window, config.window);
  std::vector<unsigned char> forward_ok;
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(previous, current, features, result.points,
                           forward_ok, error, window, config.pyramid_levels);

  // The same call the other way round. The tracker's own status flag is not
  // enough: at a 40 pixel shift it keeps 197 of 200 features while being 38.8
  // pixels wrong. Tracking back and checking we return is our own test.
  std::vector<cv::Point2f> back;
  std::vector<unsigned char> back_ok;
  cv::calcOpticalFlowPyrLK(current, previous, result.points, back, back_ok,
                           error, window, config.pyramid_levels);

  result.ok.resize(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    const cv::Point2f drift = back[i] - features[i];
    const double distance = std::sqrt(drift.x * drift.x + drift.y * drift.y);
    result.ok[i] = forward_ok[i] && back_ok[i] &&
                   distance < config.max_forward_backward &&
                   result.points[i].x >= 0 && result.points[i].y >= 0 &&
                   result.points[i].x < current.cols &&
                   result.points[i].y < current.rows;
  }
  return result;
}

}  // namespace lvio