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

}  // namespace
}  // namespace lvio
