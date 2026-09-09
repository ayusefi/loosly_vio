#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>

#include "lvio/vision/feature_tracker.h"

namespace lvio {
namespace {

// Grey everywhere, with a checkerboard painted into one rectangle, so we can
// choose exactly where the texture is.
cv::Mat Checkerboard(int width, int height, cv::Rect where, int square) {
  cv::Mat image(height, width, CV_8UC1, cv::Scalar(128));
  for (int y = where.y; y < where.y + where.height; ++y) {
    for (int x = where.x; x < where.x + where.width; ++x) {
      image.at<unsigned char>(y, x) =
          (((x / square) + (y / square)) % 2) ? 255 : 0;
    }
  }
  return image;
}

// A board in the bottom-left cell only. Row 3, column 0, so 3 * 6 + 0 = 18.
constexpr int kCrowdedCell = 18;
cv::Mat OneCrowdedCell() {
  return Checkerboard(752, 480, cv::Rect(0, 360, 125, 120), 8);
}

TEST(CellCounts, PutsEachCornerOfTheImageInItsOwnCell) {
  DetectorConfig config;
  config.grid_cols = 2;
  config.grid_rows = 2;
  const std::vector<cv::Point2f> corners = {
      {0.0F, 0.0F}, {99.0F, 0.0F}, {0.0F, 99.0F}, {99.0F, 99.0F}};

  const std::vector<int> counts = CellCounts(corners, 100, 100, config);

  EXPECT_EQ(counts.size(), 4U);
  EXPECT_EQ(counts[0], 1);
  EXPECT_EQ(counts[1], 1);
  EXPECT_EQ(counts[2], 1);
  EXPECT_EQ(counts[3], 1);
}

TEST(DetectFeatures, ABlankImageGivesNothing) {
  const cv::Mat blank(480, 752, CV_8UC1, cv::Scalar(128));
  EXPECT_TRUE(DetectFeatures(blank, DetectorConfig()).empty());
}

TEST(DetectFeatures, NoCellExceedsItsQuota) {
  const DetectorConfig config;
  const std::vector<int> counts =
      CellCounts(DetectFeatures(OneCrowdedCell(), config), 752, 480, config);
  EXPECT_EQ(*std::max_element(counts.begin(), counts.end()),
            config.max_per_cell);
}

// We cap, we do not fill.
TEST(DetectFeatures, AQuotaIsACeilingNotAFloor) {
  const DetectorConfig config;
  const std::vector<int> counts =
      CellCounts(DetectFeatures(OneCrowdedCell(), config), 752, 480, config);
  EXPECT_EQ(std::count(counts.begin(), counts.end(), 0), 23);
}

// A quota larger than the pool must change nothing at all.
TEST(DetectFeatures, AGenerousQuotaKeepsEverything) {
  const cv::Mat image = OneCrowdedCell();
  DetectorConfig config;
  config.max_per_cell = 100000;

  const std::vector<int> counts =
      CellCounts(DetectFeatures(image, config), 752, 480, config);
  EXPECT_GT(counts[kCrowdedCell], 100);
}

// Because the detector sorts best-first, the survivors in a full cell must be
// the pool's first max_per_cell entries for that cell -- the best ones.
TEST(DetectFeatures, KeepsTheBestInACellRatherThanAnArbitraryFew) {
  const cv::Mat image = OneCrowdedCell();
  const DetectorConfig config;

  DetectorConfig unlimited = config;
  unlimited.max_per_cell = 100000;
  const std::vector<cv::Point2f> pool = DetectFeatures(image, unlimited);
  const std::vector<cv::Point2f> kept = DetectFeatures(image, config);

  ASSERT_GT(pool.size(), kept.size());
  ASSERT_EQ(kept.size(), static_cast<std::size_t>(config.max_per_cell));
  for (std::size_t i = 0; i < kept.size(); ++i) {
    EXPECT_EQ(kept[i].x, pool[i].x);
    EXPECT_EQ(kept[i].y, pool[i].y);
  }
}

}  // namespace
}  // namespace lvio