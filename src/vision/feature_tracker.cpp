#include "lvio/vision/feature_tracker.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

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
  return DetectMoreFeatures(image, {}, config);
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
    result.ok[i] =
        forward_ok[i] && back_ok[i] && distance < config.max_forward_backward &&
        result.points[i].x >= 0 && result.points[i].y >= 0 &&
        result.points[i].x < current.cols && result.points[i].y < current.rows;
  }
  return result;
}

std::vector<cv::Point2f> DetectMoreFeatures(
    const cv::Mat& image, const std::vector<cv::Point2f>& existing,
    const DetectorConfig& config) {
  // Rule 1: an existing feature blocks its own neighbourhood. White means the
  // detector may look here; the black discs are the places it may not.
  cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
  for (const cv::Point2f& p : existing) {
    cv::circle(mask, p, config.min_distance, cv::Scalar(0), -1);
  }

  std::vector<cv::Point2f> pool;
  cv::goodFeaturesToTrack(image, pool, 0, config.quality_level,
                          config.min_distance, mask);

  const std::size_t cells =
      static_cast<std::size_t>(config.grid_cols * config.grid_rows);
  std::vector<int> used(cells, 0);

  // Rule 2: an existing feature counts against its cell's quota.
  // It is what keeps the whole set spread instead of only the features added
  // today.
  for (const cv::Point2f& p : existing) {
    used[static_cast<std::size_t>(
        CellIndex(p, image.cols, image.rows, config))] += 1;
  }

  std::vector<cv::Point2f> kept;
  kept.reserve(pool.size());
  for (const cv::Point2f& p : pool) {
    const std::size_t cell =
        static_cast<std::size_t>(CellIndex(p, image.cols, image.rows, config));
    if (used[cell] < config.max_per_cell) {
      used[cell] += 1;
      kept.push_back(p);
    }
  }
  return kept;
}

FeatureTracker::FeatureTracker(const DetectorConfig& detector,
                               const TrackerConfig& tracker, int min_tracks)
    : detector_(detector), tracker_(tracker), min_tracks_(min_tracks) {}

const std::vector<FeatureTrack>& FeatureTracker::Update(const cv::Mat& image) {
  if (!previous_.empty()) {
    // TrackFeatures returns one slot per input feature in the same order, so
    // slot k belongs to track k and the id needs no matching to carry across.
    std::vector<cv::Point2f> points;
    points.reserve(tracks_.size());
    for (const FeatureTrack& t : tracks_) {
      points.push_back(t.point);
    }
    const TrackResult moved = TrackFeatures(previous_, image, points, tracker_);

    std::vector<FeatureTrack> alive;
    alive.reserve(tracks_.size());
    for (std::size_t k = 0; k < tracks_.size(); ++k) {
      if (moved.ok[k]) {
        alive.push_back({tracks_[k].id, moved.points[k], tracks_[k].age + 1});
      } else {
        finished_.push_back(tracks_[k].age);
      }
    }
    tracks_.swap(alive);
  }

  if (static_cast<int>(tracks_.size()) < min_tracks_) {
    std::vector<cv::Point2f> held;
    held.reserve(tracks_.size());
    for (const FeatureTrack& t : tracks_) {
      held.push_back(t.point);
    }
    for (const cv::Point2f& p : DetectMoreFeatures(image, held, detector_)) {
      tracks_.push_back({next_id_++, p, 1});
    }
  }

  previous_ = image;
  return tracks_;
}

}  // namespace lvio