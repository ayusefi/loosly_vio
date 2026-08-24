#include <gtest/gtest.h>

#include "lvio/math/numeric_jacobian.h"
#include "lvio/math/pose.h"

namespace lvio {
namespace {

Pose MakePose() { return Pose(Exp(Vec3(0.1, 0.2, 0.3)), Vec3(1.5, -0.5, 2.0)); }

TEST(Perturb, ByZeroChangesNothing) {
  const Pose pose = MakePose();
  const Pose same = Perturb(pose, Vec6::Zero());

  EXPECT_LT((same.R - pose.R).norm(), 1e-15);
  EXPECT_LT((same.t - pose.t).norm(), 1e-15);
}

// The first three numbers must move only the position.
TEST(Perturb, TranslationHalfLeavesTheRotationAlone) {
  const Pose pose = MakePose();
  Vec6 delta = Vec6::Zero();
  delta.head<3>() = Vec3(0.1, -0.2, 0.3);

  const Pose moved = Perturb(pose, delta);

  EXPECT_LT((moved.R - pose.R).norm(), 1e-15);
  EXPECT_LT((moved.t - pose.t - Vec3(0.1, -0.2, 0.3)).norm(), 1e-15);
}

// The last three must move only the rotation.
TEST(Perturb, RotationHalfLeavesThePositionAlone) {
  const Pose pose = MakePose();
  Vec6 delta = Vec6::Zero();
  delta.tail<3>() = Vec3(0.05, 0.0, -0.02);

  const Pose turned = Perturb(pose, delta);

  EXPECT_LT((turned.t - pose.t).norm(), 1e-15);
  EXPECT_GT((turned.R - pose.R).norm(), 0.01);
}

// Lesson 17 part 5: identity on the left, -R Hat(p) on the right.
TEST(PoseJacobian, MatchesTheAnalyticFormForTransformingAPoint) {
  const Pose pose = MakePose();
  const Vec3 p(2.0, 1.0, 4.0);

  const Mat36 numeric =
      NumericJacobianPose([&p](const Pose& q) { return q * p; }, pose);

  EXPECT_LT((numeric.leftCols<3>() - Mat3::Identity()).norm(), 1e-7);
  EXPECT_LT((numeric.rightCols<3>() + pose.R * Hat(p)).norm(), 1e-7);
}

// A second, independent function, for the reason given in lesson 15.
TEST(PoseJacobian, MatchesTheAnalyticFormForTheInverseTransform) {
  const Pose pose = MakePose();
  const Vec3 v(-0.7, 1.4, 0.2);

  const Mat36 numeric = NumericJacobianPose(
      [&v](const Pose& q) { return q.Inverse() * v; }, pose);

  const Mat3 expected_left = -pose.R.transpose();
  const Mat3 expected_right = Hat(pose.R.transpose() * (v - pose.t));

  EXPECT_LT((numeric.leftCols<3>() - expected_left).norm(), 1e-7);
  EXPECT_LT((numeric.rightCols<3>() - expected_right).norm(), 1e-7);
}

// If Perturb ever switched to choice B, the left block would become R.
TEST(PoseJacobian, WouldNoticeIfWeSwitchedToTheOtherConvention) {
  const Pose pose = MakePose();
  const Vec3 p(2.0, 1.0, 4.0);

  const Mat36 numeric =
      NumericJacobianPose([&p](const Pose& q) { return q * p; }, pose);

  EXPECT_GT((numeric.leftCols<3>() - pose.R).norm(), 0.1);
}

}  // namespace
}  // namespace lvio
