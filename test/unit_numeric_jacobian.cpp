#include <gtest/gtest.h>

#include "lvio/math/numeric_jacobian.h"

namespace lvio {
namespace {

// The analytic answer we derived by hand.
TEST(NumericJacobian, AgreesWithTheAnalyticFormForRotatingAPoint) {
  const Mat3 R = Exp(Vec3(0.3, -0.5, 0.8));
  const Vec3 p(1.0, 2.0, 3.0);

  const Mat3 numeric =
      NumericJacobianSo3([&p](const Mat3& q) { return Vec3(q * p); }, R);
  const Mat3 analytic = -R * Hat(p);

  EXPECT_LT((numeric - analytic).norm(), 1e-7);
}

// The bug that a sign flip that never crashes. This is the whole reason the
// checker exists.
TEST(NumericJacobian, CatchesAFlippedSign) {
  const Mat3 R = Exp(Vec3(0.3, -0.5, 0.8));
  const Vec3 p(1.0, 2.0, 3.0);

  const Mat3 numeric =
      NumericJacobianSo3([&p](const Mat3& q) { return Vec3(q * p); }, R);
  const Mat3 wrong = R * Hat(p);

  EXPECT_GT((numeric - wrong).norm(), 1.0);
}

// A second, independent function, so the checker is not tuned to one case.
// f(R) = R^T v has the analytic Jacobian Hat(R^T v).
TEST(NumericJacobian, AgreesForTheInverseRotationOfAPoint) {
  const Mat3 R = Exp(Vec3(0.3, -0.5, 0.8));
  const Vec3 v(-0.7, 1.4, 0.2);

  const Mat3 numeric = NumericJacobianSo3(
      [&v](const Mat3& q) { return Vec3(q.transpose() * v); }, R);
  const Mat3 analytic = Hat(R.transpose() * v);

  EXPECT_LT((numeric - analytic).norm(), 1e-7);
}

// At the identity there is nothing to rotate away, so the Jacobian of R p
// reduces to -Hat(p).
TEST(NumericJacobian, ReducesToHatAtTheIdentity) {
  const Vec3 p(1.0, 2.0, 3.0);

  const Mat3 numeric = NumericJacobianSo3(
      [&p](const Mat3& q) { return Vec3(q * p); }, Mat3::Identity());

  EXPECT_LT((numeric + Hat(p)).norm(), 1e-7);
}

}  // namespace
}  // namespace lvio