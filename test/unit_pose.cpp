#include <gtest/gtest.h>

#include "lvio/math/pose.h"

namespace lvio {
namespace {

constexpr double kTol = 1e-12;

Pose MakeA() { return Pose(Exp(Vec3(0.1, 0.2, 0.3)), Vec3(1.5, -0.5, 2.0)); }
Pose MakeB() { return Pose(Exp(Vec3(-0.2, 0.1, 0.4)), Vec3(0.5, -1.0, 2.0)); }

TEST(Pose, DefaultDoesNothing) {
  const Pose identity;
  const Vec3 p(2.0, 1.0, 4.0);

  EXPECT_LT((identity * p - p).norm(), kTol);
}

// Composing must equal doing it in two steps.
TEST(Pose, CompositionEqualsApplyingBothInTurn) {
  const Pose a = MakeA();
  const Pose b = MakeB();
  const Vec3 p(2.0, 1.0, 4.0);

  const Vec3 two_steps = a * (b * p);
  const Vec3 composed = (a * b) * p;

  EXPECT_LT((two_steps - composed).norm(), kTol);
}

TEST(Pose, InverseUndoesThePose) {
  const Pose a = MakeA();
  const Pose round_trip = a.Inverse() * a;

  EXPECT_LT((round_trip.R - Mat3::Identity()).norm(), kTol);
  EXPECT_LT(round_trip.t.norm(), kTol);
}

// The translation of the inverse is -R^T t, not -t. This test
// fails loudly for the common mistake.
TEST(Pose, InverseTranslationIsNotJustNegated) {
  const Pose a = MakeA();

  EXPECT_GT((a.Inverse().t + a.t).norm(), 0.1);
}

// Lesson 16, part 6.
TEST(Pose, BetweenTakesYouFromTheFirstToTheSecond) {
  const Pose a = MakeA();
  const Pose b = MakeB();

  const Pose moved = a * Between(a, b);

  EXPECT_LT((moved.R - b.R).norm(), kTol);
  EXPECT_LT((moved.t - b.t).norm(), kTol);
}

// Move the whole world somewhere else. The relative pose must not notice.
TEST(Pose, BetweenDoesNotDependOnTheWorldOrigin) {
  const Pose a = MakeA();
  const Pose b = MakeB();
  const Pose shift(Exp(Vec3(0.7, -0.3, 1.1)), Vec3(-4.0, 9.0, 2.5));

  const Pose direct = Between(a, b);
  const Pose after_shift = Between(shift * a, shift * b);

  EXPECT_LT((direct.R - after_shift.R).norm(), kTol);
  EXPECT_LT((direct.t - after_shift.t).norm(), kTol);
}

TEST(Interpolate, EndsMatchTheInputs) {
  const Pose a = MakeA();
  const Pose b = MakeB();

  const Pose at_zero = Interpolate(a, b, 0.0);
  const Pose at_one = Interpolate(a, b, 1.0);

  EXPECT_LT((at_zero.R - a.R).norm(), kTol);
  EXPECT_LT((at_zero.t - a.t).norm(), kTol);
  EXPECT_LT((at_one.R - b.R).norm(), kTol);
  EXPECT_LT((at_one.t - b.t).norm(), kTol);
}

// Lesson 1: averaging rotations does not give a rotation. This must.
TEST(Interpolate, AlwaysProducesARotation) {
  const Pose a = MakeA();
  const Pose b = MakeB();

  for (double s : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
    const Pose mid = Interpolate(a, b, s);

    EXPECT_LT((mid.R.transpose() * mid.R - Mat3::Identity()).norm(), 1e-14)
        << "at s = " << s;
    EXPECT_NEAR(mid.R.determinant(), 1.0, 1e-14) << "at s = " << s;
  }
}

// A fraction s of the way must be s of the angle, not merely somewhere between.
TEST(Interpolate, CoversTheAngleProportionally) {
  const Pose a = MakeA();
  const Pose b = MakeB();
  const double total = Log(a.R.transpose() * b.R).norm();

  for (double s : {0.25, 0.5, 0.75}) {
    const Pose mid = Interpolate(a, b, s);
    const double covered = Log(a.R.transpose() * mid.R).norm();

    EXPECT_NEAR(covered, s * total, 1e-12) << "at s = " << s;
  }
}

TEST(Interpolate, MovesThePositionInAStraightLine) {
  const Pose a = MakeA();
  const Pose b = MakeB();

  const Pose mid = Interpolate(a, b, 0.5);

  EXPECT_LT((mid.t - 0.5 * (a.t + b.t)).norm(), kTol);
}

// Averaging the rotation matrices, which is what Lesson 1 warned against.
TEST(Interpolate, BeatsAveragingTheMatrices) {
  const Pose a = MakeA();
  const Pose b = MakeB();

  const Mat3 averaged = 0.5 * (a.R + b.R);

  EXPECT_GT((averaged.transpose() * averaged - Mat3::Identity()).norm(), 1e-3);
}
}  // namespace
}  // namespace lvio
