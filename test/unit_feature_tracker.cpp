#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>

#include "lvio/vision/feature_tracker.h"
#include <opencv2/imgproc.hpp>

#include <numeric>

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

// Random shapes at random sizes and shades. Unlike a checkerboard this does not
// repeat, so a coarse pyramid level cannot match the wrong copy of anything.
cv::Mat Poster() {
  cv::Mat image(480, 752, CV_8UC1, cv::Scalar(120));
  cv::RNG rng(11);
  for (int i = 0; i < 220; ++i) {
    const int x = rng.uniform(0, 752);
    const int y = rng.uniform(0, 480);
    const int size = rng.uniform(6, 34);
    const cv::Scalar shade(rng.uniform(0, 256));
    if (i % 2) {
      cv::rectangle(image, cv::Rect(x, y, size, rng.uniform(6, 34)), shade, -1);
    } else {
      cv::circle(image, cv::Point(x, y), size / 2, shade, -1);
    }
  }
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.8);
  return image;
}

// Features away from the edges, so shifting the image cannot push a window out.
std::vector<cv::Point2f> PosterFeatures(const cv::Mat& image) {
  std::vector<cv::Point2f> all;
  cv::goodFeaturesToTrack(image, all, 300, 0.01, 20);
  std::vector<cv::Point2f> inner;
  for (const cv::Point2f& p : all) {
    if (p.x > 60 && p.x < 692 && p.y > 40 && p.y < 440) {
      inner.push_back(p);
    }
  }
  return inner;
}

cv::Mat ShiftedRight(const cv::Mat& image, int pixels) {
  const cv::Mat move =
      (cv::Mat_<float>(2, 3) << 1, 0, static_cast<float>(pixels), 0, 1, 0);
  cv::Mat out;
  cv::warpAffine(image, out, move, image.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  return out;
}

int Survivors(const TrackResult& result) {
  return static_cast<int>(std::count(result.ok.begin(), result.ok.end(), 1));
}

TEST(TrackFeatures, AnEmptyInputGivesAnEmptyResultRatherThanThrowing) {
  const cv::Mat image = Poster();
  const TrackResult result = TrackFeatures(image, image, {}, TrackerConfig());
  EXPECT_TRUE(result.points.empty());
  EXPECT_TRUE(result.ok.empty());
}

TEST(TrackFeatures, TheResultKeepsOneSlotPerInputFeature) {
  const cv::Mat image = Poster();
  const std::vector<cv::Point2f> features = PosterFeatures(image);
  ASSERT_GT(features.size(), 100U);

  const TrackResult result =
      TrackFeatures(image, image, features, TrackerConfig());
  EXPECT_EQ(result.points.size(), features.size());
  EXPECT_EQ(result.ok.size(), features.size());
}

TEST(TrackFeatures, AnIdenticalPairDoesNotMove) {
  const cv::Mat image = Poster();
  const std::vector<cv::Point2f> features = PosterFeatures(image);
  const TrackResult result =
      TrackFeatures(image, image, features, TrackerConfig());

  EXPECT_EQ(Survivors(result), static_cast<int>(features.size()));
  for (std::size_t i = 0; i < features.size(); ++i) {
    EXPECT_NEAR(result.points[i].x, features[i].x, 1e-3);
    EXPECT_NEAR(result.points[i].y, features[i].y, 1e-3);
  }
}

TEST(TrackFeatures, RecoversAShiftThatFitsInsideTheWindow) {
  const cv::Mat image = Poster();
  const std::vector<cv::Point2f> features = PosterFeatures(image);
  const TrackResult result =
      TrackFeatures(image, ShiftedRight(image, 7), features, TrackerConfig());

  EXPECT_GT(Survivors(result), static_cast<int>(features.size()) * 9 / 10);
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (result.ok[i]) {
      EXPECT_NEAR(result.points[i].x - features[i].x, 7.0, 0.05);
      EXPECT_NEAR(result.points[i].y - features[i].y, 0.0, 0.05);
    }
  }
}

// The window half-width is 10 pixels, so 30 is far outside it. Only the
// pyramid can cross that gap.
TEST(TrackFeatures, ThePyramidCrossesAShiftTheWindowCannot) {
  const cv::Mat image = Poster();
  const cv::Mat shifted = ShiftedRight(image, 30);
  const std::vector<cv::Point2f> features = PosterFeatures(image);

  TrackerConfig single_level;
  single_level.pyramid_levels = 0;

  const TrackResult pyramid =
      TrackFeatures(image, shifted, features, TrackerConfig());
  const TrackResult flat =
      TrackFeatures(image, shifted, features, single_level);

  EXPECT_GT(Survivors(pyramid), static_cast<int>(features.size()) * 8 / 10);
  EXPECT_LT(Survivors(flat), static_cast<int>(features.size()) / 5);

  for (std::size_t i = 0; i < features.size(); ++i) {
    if (pyramid.ok[i]) {
      EXPECT_NEAR(pyramid.points[i].x - features[i].x, 30.0, 0.05);
    }
  }
}

}  // namespace
}  // namespace lvio