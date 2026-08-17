#include "lvio/math/so3.h"

#include <gtest/gtest.h>

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

TEST(Hat, OfZeroIsZero) {
  EXPECT_LT(Hat(Vec3::Zero()).norm(), kTol);
}

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
  Mat3 R = Exp(Vec3(0.0, 0.0, M_PI/2.0));

  Mat3 expected;
  expected << 0.0, -1.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 0.0, 1.0;

  EXPECT_LT((R - expected).norm(), 1e-15);
}

// A rotation never change the length -> R^T R = I and det(R) = 1
TEST(Exp, AlwaysProducesARotation) {
  const Vec3 angles[] = {Vec3(1e-12, 0.0, 0.0), Vec3(0.001, -0.002, 0.003),
                         Vec3(0.3, -0.7, 1.2),  Vec3(2.0, 2.0, 2.0)};

  for (const Vec3& angle : angles) {
    Mat3 R = Exp(angle);

    EXPECT_LT((R.transpose() * R - Mat3::Identity()).norm(), 1e-14);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-14);
  }
}

// The two branches must agree where they meet.
TEST(Exp, BranchesAgreeAtTheThreshold) {
  const Vec3 axis = Vec3(0.26726124, 0.53452248, 0.80178373);
  const Mat3 just_below = Exp(axis * (kSmallAngle * 0.999));
  const Mat3 just_above = Exp(axis * (kSmallAngle * 1.001));

  EXPECT_LT((just_below - just_above).norm(), 1e-4);
}

}  // namespace
}  // namespace lvio
