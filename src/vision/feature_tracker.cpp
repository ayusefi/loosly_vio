#include "lvio/vision/feature_tracker.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

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

}  // namespace lvio