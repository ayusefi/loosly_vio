#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace lvio {

// How many corners to keep, and how evenly to spread them. The grid is what
// stops one textured corner of the room from taking every slot.
struct DetectorConfig {
  int grid_cols = 6;
  int grid_rows = 4;
  int max_per_cell = 25;
  double quality_level = 0.01;
  double min_distance = 10.0;
};

// Corners, capped per grid cell. The cap is a ceiling, not a floor: a cell with
// nothing worth detecting stays empty.
std::vector<cv::Point2f> DetectFeatures(const cv::Mat& image,
                                        const DetectorConfig& config);

// How many features fell in each cell, row-major, grid_rows * grid_cols long.
// This is the spread statistic, and the tests measure it.
std::vector<int> CellCounts(const std::vector<cv::Point2f>& features, int width,
                            int height, const DetectorConfig& config);

// How to follow a feature into the next image.
struct TrackerConfig {
  int window = 21;
  int pyramid_levels = 3;
  double max_forward_backward = 1.0;
};

// One entry per input feature, in the same order, so the caller's own track
// bookkeeping stays aligned. A rejected track keeps its slot with ok = 0.
// This is unsigned char rather than bool because std::vector<bool> is not a
// container of bool, and OpenCV wants to write into a real byte array.
struct TrackResult {
  std::vector<cv::Point2f> points;
  std::vector<unsigned char> ok;
};

// Follows each feature from `previous` into `current`, then tracks the answer
// back again and rejects anything that fails to return where it started.
TrackResult TrackFeatures(const cv::Mat& previous, const cv::Mat& current,
                          const std::vector<cv::Point2f>& features,
                          const TrackerConfig& config);

// A corner followed through time. The id never changes for as long as the track
// lives, so the same piece of world keeps one name across frames. It is a name,
// not a position in any array: the arrays are compacted every frame.
struct FeatureTrack {
  int id = 0;
  cv::Point2f point;
  int age = 0;
};

// Detects corners while avoiding the ones we already hold. Existing features
// block their own neighbourhood and count against their cell's quota, so the
// whole set stays spread rather than just the new arrivals.
std::vector<cv::Point2f> DetectMoreFeatures(
    const cv::Mat& image, const std::vector<cv::Point2f>& existing,
    const DetectorConfig& config);

// Keeps a set of corners alive across frames: tracks them forward, drops the
// ones that fail the round trip, and tops up when too few are left.
class FeatureTracker {
 public:
  FeatureTracker(const DetectorConfig& detector, const TrackerConfig& tracker,
                 int min_tracks);

  // Feeds one image and returns the tracks alive in it. The first call only
  // detects, because there is nothing to track from.
  const std::vector<FeatureTrack>& Update(const cv::Mat& image);

  const std::vector<FeatureTrack>& Tracks() const { return tracks_; }

  // The length of every track that has already died. Add the ages of the still
  // living tracks to get the full histogram.
  const std::vector<int>& FinishedLengths() const { return finished_; }

 private:
  DetectorConfig detector_;
  TrackerConfig tracker_;
  int min_tracks_;
  cv::Mat previous_;
  std::vector<FeatureTrack> tracks_;
  std::vector<int> finished_;
  int next_id_ = 0;
};

}  // namespace lvio