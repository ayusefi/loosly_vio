#include "lvio/math/so3.h"

#include <gtest/gtest.h>

namespace lvio {
namespace {

constexpr double kTol = 1e-15;

// Lesson 2, part 6: the general shape of a skew-symmetric matrix.
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

// Lesson 2, part 4: this is what Hat is for, and the test that catches a sign
// error anywhere in the layout.
TEST(Hat, ReproducesTheCrossProduct) {
  const Vec3 a(0.4, -1.3, 2.7);
  const Vec3 b(-0.8, 0.5, 1.1);

  EXPECT_LT((Hat(a) * b - a.cross(b)).norm(), kTol);
}

// Lesson 3: the shape is forced by the length test, S^T = -S.
TEST(Hat, IsSkewSymmetric) {
  const Vec3 v(0.7, -0.2, 1.4);
  const Mat3 m = Hat(v);

  EXPECT_LT((m.transpose() + m).norm(), kTol);
  EXPECT_DOUBLE_EQ(m.trace(), 0.0);
}

// Lesson 2, part 6: a turn about x fills the first slot, y the second, z the
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

}  // namespace
}  // namespace lvio
