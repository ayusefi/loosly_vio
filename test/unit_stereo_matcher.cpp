#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

#include "lvio/vision/stereo_matcher.h"

namespace lvio {
namespace {

// A camera with round numbers and no lens distortion, so the test checks the
// stereo maths and nothing else.
Camera Ideal() {
  Camera c;
  c.width = 752;
  c.height = 480;
  c.fu = 400.0;
  c.fv = 400.0;
  c.cu = 376.0;
  c.cv = 240.0;
  return c;
}

constexpr double kBaseline = 0.11;

StereoCalibration IdealPair() {
  StereoCalibration s;
  s.left = Ideal();
  s.right = Ideal();
  s.T_left_right.R = Mat3::Identity();
  s.T_left_right.t = Vec3(kBaseline, 0.0, 0.0);
  return s;
}

// The same poster as the tracker tests: distinctive, non-repeating texture.
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

// With identity rotation and a pure-x baseline, a flat wall at constant depth
// produces one horizontal disparity f*b/z everywhere. So the right image of a
// wall is just the left image slid sideways -- an exact synthetic stereo pair.
cv::Mat RightImageOfAWallAt(const cv::Mat& left, double depth) {
  const double disparity = Ideal().fu * kBaseline / depth;
  const cv::Mat move =
      (cv::Mat_<float>(2, 3) << 1, 0, static_cast<float>(-disparity), 0, 1, 0);
  cv::Mat out;
  cv::warpAffine(left, out, move, left.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  return out;
}

std::vector<FeatureTrack> TracksAwayFromTheBorder(const cv::Mat& image) {
  std::vector<cv::Point2f> found;
  cv::goodFeaturesToTrack(image, found, 300, 0.01, 20);
  std::vector<FeatureTrack> tracks;
  int id = 0;
  for (const cv::Point2f& p : found) {
    if (p.x > 60 && p.x < 692 && p.y > 40 && p.y < 440) {
      tracks.push_back({id++, p, 1});
    }
  }
  return tracks;
}

TEST(TriangulateMidpoint, RecoversAPointFromExactRays) {
  const Vec3 baseline(kBaseline, 0.0, 0.0);
  for (const double depth : {1.0, 5.0, 20.0}) {
    const Vec3 truth(0.3, -0.2, depth);
    const Vec3 left = truth.normalized();
    const Vec3 right = (truth - baseline).normalized();

    const std::optional<Vec3> got =
        TriangulateMidpoint(left, right, baseline);
    ASSERT_TRUE(got.has_value());
    EXPECT_LT((*got - truth).norm(), 1e-9);
  }
}

TEST(TriangulateMidpoint, RefusesParallelRays) {
  const Vec3 forward(0.0, 0.0, 1.0);
  EXPECT_FALSE(
      TriangulateMidpoint(forward, forward, Vec3(kBaseline, 0.0, 0.0))
          .has_value());
}

// Scaling a ray must not move the point: only the direction carries meaning.
TEST(TriangulateMidpoint, DoesNotCareAboutRayLength) {
  const Vec3 baseline(kBaseline, 0.0, 0.0);
  const Vec3 truth(0.1, 0.2, 4.0);
  const Vec3 left = truth.normalized();
  const Vec3 right = (truth - baseline).normalized();

  const std::optional<Vec3> unit = TriangulateMidpoint(left, right, baseline);
  const std::optional<Vec3> scaled =
      TriangulateMidpoint(left * 7.0, right * 0.3, baseline);
  ASSERT_TRUE(unit.has_value());
  ASSERT_TRUE(scaled.has_value());
  EXPECT_LT((*unit - *scaled).norm(), 1e-9);
}

TEST(MatchStereo, AnEmptyTrackListGivesNoPoints) {
  const cv::Mat image = Poster();
  EXPECT_TRUE(MatchStereo(image, image, {}, IdealPair(),
                          StereoMatcherConfig())
                  .empty());
}

TEST(MatchStereo, RecoversTheDepthOfASyntheticWall) {
  const cv::Mat left = Poster();
  const std::vector<FeatureTrack> tracks = TracksAwayFromTheBorder(left);
  ASSERT_GT(tracks.size(), 50U);

  for (const double depth : {2.0, 5.0, 10.0}) {
    const std::vector<StereoPoint> points =
        MatchStereo(left, RightImageOfAWallAt(left, depth), tracks,
                    IdealPair(), StereoMatcherConfig());
    ASSERT_GT(points.size(), tracks.size() / 2);

    std::vector<double> depths;
    for (const StereoPoint& p : points) {
      depths.push_back(p.point.z());
    }
    std::sort(depths.begin(), depths.end());
    // 1% of the true depth. Measured worst case is about 0.6%, and it is KLT's
    // sub-pixel accuracy rather than the triangulation.
    EXPECT_NEAR(depths[depths.size() / 2], depth, depth * 0.01);
  }
}

// The whole point of part 7: a wall beyond max_depth must be thrown away.
TEST(MatchStereo, RejectsAWallBeyondMaxDepth) {
  const cv::Mat left = Poster();
  const std::vector<FeatureTrack> tracks = TracksAwayFromTheBorder(left);

  StereoMatcherConfig config;
  config.max_depth = 4.0;

  const std::vector<StereoPoint> near_wall = MatchStereo(
      left, RightImageOfAWallAt(left, 2.0), tracks, IdealPair(), config);
  const std::vector<StereoPoint> far_wall = MatchStereo(
      left, RightImageOfAWallAt(left, 40.0), tracks, IdealPair(), config);

  EXPECT_GT(near_wall.size(), tracks.size() / 2);
  EXPECT_LT(far_wall.size(), tracks.size() / 10);
}

// A point keeps the name its track gave it, so later lessons can join a 3D
// point in this frame to the same piece of world in the next one.
TEST(MatchStereo, CarriesTheTrackIdOntoThePoint) {
  const cv::Mat left = Poster();
  const std::vector<FeatureTrack> tracks = TracksAwayFromTheBorder(left);
  const std::vector<StereoPoint> points =
      MatchStereo(left, RightImageOfAWallAt(left, 3.0), tracks, IdealPair(),
                  StereoMatcherConfig());

  ASSERT_FALSE(points.empty());
  for (const StereoPoint& p : points) {
    const auto same = std::find_if(
        tracks.begin(), tracks.end(),
        [&](const FeatureTrack& t) { return t.id == p.id; });
    ASSERT_NE(same, tracks.end());
    EXPECT_FLOAT_EQ(same->point.x, p.left.x);
    EXPECT_FLOAT_EQ(same->point.y, p.left.y);
  }
}

}  // namespace
}  // namespace lvio