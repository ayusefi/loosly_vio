// Smoke test for the dependency wiring. Every library the project relies on is
// exercised here, so a broken toolchain fails in one obvious place instead of
// halfway through a sequence.

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

#include "lvio/common/build_info.h"

TEST(Toolchain, BuildSummaryIsPopulated) {
  const std::string summary = lvio::BuildSummary();
  EXPECT_NE(summary.find("lvio"), std::string::npos);
}

TEST(Toolchain, EigenSolvesALinearSystem) {
  Eigen::Matrix3d a;
  a << 2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0;
  const Eigen::Vector3d b(1.0, 2.0, 3.0);

  const Eigen::Vector3d x = a.ldlt().solve(b);

  EXPECT_LT((a * x - b).norm(), 1e-12);
}

TEST(Toolchain, SophusRoundTripsSe3) {
  Sophus::Vector6d tangent;
  tangent << 0.1, -0.2, 0.3, 0.4, 0.5, -0.6;

  const Sophus::SE3d pose = Sophus::SE3d::exp(tangent);

  EXPECT_LT((pose.log() - tangent).norm(), 1e-12);
  EXPECT_LT((pose.rotationMatrix() * pose.rotationMatrix().transpose() -
             Eigen::Matrix3d::Identity())
                .norm(),
            1e-12);
}

TEST(Toolchain, OpenCvFindsCorners) {
  cv::Mat image = cv::Mat::zeros(120, 160, CV_8UC1);
  cv::rectangle(image, cv::Rect(40, 30, 60, 50), cv::Scalar(255), cv::FILLED);

  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(image, corners, 20, 0.01, 5.0);

  EXPECT_GE(corners.size(), 4u);
}

namespace {

struct QuadraticResidual {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }
};

}  // namespace

TEST(Toolchain, CeresSolvesAScalarProblem) {
  double x = 0.5;

  ceres::Problem problem;
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<QuadraticResidual, 1, 1>(
          new QuadraticResidual),
      nullptr, &x);

  ceres::Solver::Options options;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Ceres stops on a relative function tolerance of 1e-6 by default, so there
  // is no point asserting tighter than the solver's own stopping criterion.
  EXPECT_TRUE(summary.IsSolutionUsable());
  EXPECT_NEAR(x, 10.0, 1e-6);
}

TEST(Toolchain, YamlCppParsesASequence) {
  const YAML::Node node =
      YAML::Load("intrinsics: [458.654, 457.296, 367.215, 248.375]");

  const auto intrinsics = node["intrinsics"].as<std::vector<double>>();

  ASSERT_EQ(intrinsics.size(), 4u);
  EXPECT_DOUBLE_EQ(intrinsics[0], 458.654);
}

TEST(Toolchain, SpdlogHasADefaultLogger) {
  spdlog::set_level(spdlog::level::off);
  EXPECT_NE(spdlog::default_logger(), nullptr);
}
