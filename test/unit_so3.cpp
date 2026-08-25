#include <gtest/gtest.h>

#include "lvio/math/so3.h"

namespace lvio {
namespace {

constexpr double kTol = 1e-15;

// The general shape of a skew-symmetric matrix.
TEST(Hat, MatchesTheDerivedShape) {
  const Vec3 v(1.0, 2.0, 3.0);
  const Mat3 m = Hat(v);

  EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(m(0, 1), -3.0);
  EXPECT_DOUBLE_EQ(m(0, 2), 2.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(m(1, 2), -1.0);
  EXPECT_DOUBLE_EQ(m(2, 0), -2.0);
  EXPECT_DOUBLE_EQ(m(2, 1), 1.0);
  EXPECT_DOUBLE_EQ(m(2, 2), 0.0);
}

// This is what Hat is for, and the test that catches a sign
// error anywhere in the layout.
TEST(Hat, ReproducesTheCrossProduct) {
  const Vec3 a(0.4, -1.3, 2.7);
  const Vec3 b(-0.8, 0.5, 1.1);

  EXPECT_LT((Hat(a) * b - a.cross(b)).norm(), kTol);
}

// The shape is forced by the length test, S^T = -S.
TEST(Hat, IsSkewSymmetric) {
  const Vec3 v(0.7, -0.2, 1.4);
  const Mat3 m = Hat(v);

  EXPECT_LT((m.transpose() + m).norm(), kTol);
  EXPECT_DOUBLE_EQ(m.trace(), 0.0);
}

// A turn about x fills the first slot, y the second, z the
// third. Getting this wrong puts the rotation on the wrong axis.
TEST(Hat, PutsEachAxisInItsOwnSlot) {
  EXPECT_LT((Hat(Vec3::UnitX()) * Vec3::UnitY() - Vec3::UnitZ()).norm(), kTol);
  EXPECT_LT((Hat(Vec3::UnitY()) * Vec3::UnitZ() - Vec3::UnitX()).norm(), kTol);
  EXPECT_LT((Hat(Vec3::UnitZ()) * Vec3::UnitX() - Vec3::UnitY()).norm(), kTol);
}

TEST(Hat, OfZeroIsZero) { EXPECT_LT(Hat(Vec3::Zero()).norm(), kTol); }

TEST(Vee, UndoesHat) {
  const Vec3 v(-2.5, 0.9, 3.1);

  EXPECT_LT((Vee(Hat(v)) - v).norm(), kTol);
}

// No rotation must give the identity, Not NAN
TEST(Exp, ofZeroIsIdentity) {
  Mat3 R = Exp(Vec3::Zero());

  EXPECT_LT((R - Mat3::Identity()).norm(), kTol);
}

// A worked example by hand
TEST(Exp, NinetyDegreesAboutZ) {
  Mat3 R = Exp(Vec3(0.0, 0.0, M_PI / 2.0));

  Mat3 expected;
  // clang-format off
  expected << 0.0, -1.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 0.0, 1.0;
  // clang-format on

  EXPECT_LT((R - expected).norm(), 1e-15);
}

// A rotation never change the length -> R^T R = I and det(R) = 1
TEST(Exp, AlwaysProducesARotation) {
  const Vec3 angles[] = {Vec3(1e-12, 0.0, 0.0), Vec3(0.001, -0.002, 0.003),
                         Vec3(0.3, -0.7, 1.2), Vec3(2.0, 2.0, 2.0)};

  for (const Vec3& angle : angles) {
    Mat3 R = Exp(angle);

    EXPECT_LT((R.transpose() * R - Mat3::Identity()).norm(), 1e-14);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-14);
  }
}

// The two branches must not produce a jump at the threshold. The gap must be
// small enough that the function's own change across it stays well under the
// tolerance, or the test measures the slope instead of the branch.
TEST(Exp, BranchesAgreeAtTheThreshold) {
  const Vec3 axis(0.26726124, 0.53452248, 0.80178373);
  const Mat3 below = Exp(axis * (kSmallAngle * (1.0 - 1e-7)));
  const Mat3 above = Exp(axis * (kSmallAngle * (1.0 + 1e-7)));

  EXPECT_LT((below - above).norm(), 1e-8);
}

TEST(Log, ofIdentityIsZero) { EXPECT_LT(Log(Mat3::Identity()).norm(), kTol); }

TEST(Log, NinetyDegreesAboutZ) {
  Mat3 R;
  // clang-format off
  R << 0.0, -1.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 0.0, 1.0;
  // clang-format on

  EXPECT_LT((Log(R) - Vec3(0.0, 0.0, M_PI / 2.0)).norm(), 1e-15);
}

TEST(Log, UndoesExp) {
  const Vec3 cases[] = {
      Vec3(0.0, 0.0, 0.0),        Vec3(1e-12, 0.0, 0.0),
      Vec3(0.001, -0.002, 0.003), Vec3(0.005, 0.005, 0.005),
      Vec3(0.3, -0.7, 1.2),       Vec3(1.0, 1.0, 1.0),
      Vec3(0.0, 0.0, 3.0),        Vec3(-2.1, 0.4, 1.6),
  };

  for (const Vec3& w : cases) {
    EXPECT_LT((Log(Exp(w)) - w).norm(), 1e-12)
        << "failed for " << w.transpose();
  }
}

TEST(Log, HandlesHalfTurnsWithoutDemandingASign) {
  const Vec3 axes[] = {Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ(),
                       Vec3(0.0, 0.6, 0.8), Vec3(1.0, 1.0, 1.0).normalized()};

  for (const Vec3& axis : axes) {
    const Mat3 R = Exp(axis * M_PI);
    const Vec3 w = Log(R);

    EXPECT_NEAR(w.norm(), M_PI, 1e-7);
    EXPECT_LT((Exp(w) - R).norm(), 1e-7);
  }
}

TEST(Log, IsAccurateJustBelowAHalfTurn) {
  const Vec3 axis = Vec3(0.26726124, 0.53452248, 0.80178373);

  for (double gap : {1e-3, 1e-5, 1e-7}) {
    const Vec3 w = axis * (M_PI - gap);
    EXPECT_LT((Log(Exp(w)) - w).norm(), 1e-6) << "gap " << gap;
  }
}

TEST(RightJacobian, OfZeroIsIdentity) {
  EXPECT_LT((RightJacobian(Vec3::Zero()) - Mat3::Identity()).norm(), kTol);
  EXPECT_LT((RightJacobianInverse(Vec3::Zero()) - Mat3::Identity()).norm(),
            kTol);
}

TEST(RightJacobian, InverseUndoesIt) {
  const Vec3 axis(0.26726124, 0.53452248, 0.80178373);

  for (double m : {1e-8, 1e-3, 0.05, 0.5, 2.0, 3.0}) {
    const Vec3 theta = axis * m;
    const Mat3 product = RightJacobian(theta) * RightJacobianInverse(theta);

    EXPECT_LT((product - Mat3::Identity()).norm(), 1e-10)
        << "at |theta| = " << m;
  }
}

// This is the definition. The nudge is deliberately NOT
// parallel to theta, because a parallel one passes even with the wrong formula.
TEST(RightJacobian, SatisfiesItsDefiningProperty) {
  const Vec3 theta(0.4, -0.7, 1.1);
  const Vec3 delta(1e-5, -2e-5, 1.5e-5);

  const Mat3 exact = Exp(theta + delta);
  const Mat3 with_jacobian = Exp(theta) * Exp(RightJacobian(theta) * delta);

  EXPECT_LT((exact - with_jacobian).norm(), 1e-9);
}

// Proof that the test above is actually testing something: without the
// Jacobian, the same comparison fails badly.
TEST(RightJacobian, IsNeededAtAll) {
  const Vec3 theta(0.4, -0.7, 1.1);
  const Vec3 delta(1e-5, -2e-5, 1.5e-5);

  const Mat3 exact = Exp(theta + delta);
  const Mat3 naive = Exp(theta) * Exp(delta);

  EXPECT_GT((exact - naive).norm(), 1e-6);
}

// This is the definition. The nudge is deliberately NOT
// parallel to theta, because a parallel one passes even with the wrong formula.
TEST(RightJacobian, ParallelNudgesHideTheProblem) {
  const Vec3 theta(0.4, -0.7, 1.1);
  const Vec3 delta = theta.normalized() * 2e-5;

  EXPECT_LT((Exp(theta + delta) - Exp(theta) * Exp(delta)).norm(), 1e-12);
}

TEST(RightJacobian, BranchesAgreeAtTheThreshold) {
  const Vec3 axis(0.26726124, 0.53452248, 0.80178373);
  const Mat3 below = RightJacobian(axis * (kSmallAngle * (1.0 - 1e-7)));
  const Mat3 above = RightJacobian(axis * (kSmallAngle * (1.0 + 1e-7)));

  EXPECT_LT((below - above).norm(), 1e-8);
}
}  // namespace
}  // namespace lvio
