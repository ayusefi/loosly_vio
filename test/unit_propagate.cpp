#include <gtest/gtest.h>

#include "lvio/imu/propagate.h"
#include "lvio/math/so3.h"

namespace lvio {
namespace {

// A level, stationary drone. The accelerometer reads gravity, upward.
ImuSample StationaryLevelReading() {
  ImuSample s;
  s.gyro = Vec3::Zero();
  s.accel = Vec3(0.0, 0.0, 9.81);
  return s;
}

// The plan's zero-motion test: a perfect stationary IMU must not move at all.
TEST(Propagate, AStationaryDroneDoesNotMove) {
  NavState state;
  const ImuBias no_bias;
  const double dt = 0.005;

  for (int step = 0; step < 12000; ++step) {  // 60 seconds at 200 Hz
    state = Propagate(state, StationaryLevelReading(), no_bias, dt);
  }

  EXPECT_LT(state.pose.t.norm(), 1e-9);
  EXPECT_LT(state.velocity.norm(), 1e-9);
  EXPECT_LT((state.pose.R - Mat3::Identity()).norm(), 1e-12);
}

// Reading zero means nothing is holding it up, so it must fall at g.
TEST(Propagate, ReadingZeroMeansFreeFall) {
  NavState state;
  ImuSample falling;
  falling.gyro = Vec3::Zero();
  falling.accel = Vec3::Zero();
  const ImuBias no_bias;
  const double dt = 0.005;

  for (int step = 0; step < 200; ++step) {  // one second
    state = Propagate(state, falling, no_bias, dt);
  }

  // Half g t squared, downward, with t = 1.
  EXPECT_NEAR(state.pose.t.z(), -0.5 * 9.81, 0.05);
  EXPECT_NEAR(state.velocity.z(), -9.81, 0.05);
  EXPECT_LT(std::hypot(state.pose.t.x(), state.pose.t.y()), 1e-12);
}

// A constant turn rate about z for one second must give exactly that rotation.
TEST(Propagate, TurnsByTheGyroReading) {
  NavState state;
  ImuSample turning;
  turning.gyro = Vec3(0.0, 0.0, 0.7);
  turning.accel = Vec3(0.0, 0.0, 9.81);
  const ImuBias no_bias;
  const double dt = 0.005;

  for (int step = 0; step < 200; ++step) {
    state = Propagate(state, turning, no_bias, dt);
  }

  EXPECT_LT((Log(state.pose.R) - Vec3(0.0, 0.0, 0.7)).norm(), 1e-9);
}

// The bias must be removed, not added. Getting the sign wrong doubles it.
TEST(Propagate, SubtractsTheBiasRatherThanAddingIt) {
  NavState state;
  ImuSample reading = StationaryLevelReading();
  reading.gyro = Vec3(0.0, 0.0, 0.3);  // entirely bias, the drone is still

  ImuBias bias;
  bias.gyro = Vec3(0.0, 0.0, 0.3);

  for (int step = 0; step < 200; ++step) {
    state = Propagate(state, reading, bias, 0.005);
  }

  EXPECT_LT((state.pose.R - Mat3::Identity()).norm(), 1e-12);
}

// What the fifteen numbers mean.
struct Filter {
  NavState nav;
  ImuBias bias;
};

Filter ApplyError(const Filter& f, const Vec15& dx) {
  Filter out = f;
  out.nav.pose.t += dx.segment<3>(0);
  out.nav.velocity += dx.segment<3>(3);
  out.nav.pose.R = f.nav.pose.R * Exp(dx.segment<3>(6));
  out.bias.accel += dx.segment<3>(9);
  out.bias.gyro += dx.segment<3>(12);
  return out;
}

Vec15 ErrorBetween(const Filter& truth, const Filter& ours) {
  Vec15 dx = Vec15::Zero();
  dx.segment<3>(0) = truth.nav.pose.t - ours.nav.pose.t;
  dx.segment<3>(3) = truth.nav.velocity - ours.nav.velocity;
  dx.segment<3>(6) = Log(ours.nav.pose.R.transpose() * truth.nav.pose.R);
  dx.segment<3>(9) = truth.bias.accel - ours.bias.accel;
  dx.segment<3>(12) = truth.bias.gyro - ours.bias.gyro;
  return dx;
}

Filter MakeFilter() {
  Filter f;
  f.nav.pose = Pose(Exp(Vec3(0.2, -0.4, 0.7)), Vec3(1.5, -0.5, 2.0));
  f.nav.velocity = Vec3(0.6, 1.1, -0.3);
  f.bias.accel = Vec3(-0.025, 0.137, 0.076);
  f.bias.gyro = Vec3(-0.003, 0.021, 0.079);
  return f;
}

ImuSample MakeSample() {
  ImuSample s;
  s.gyro = Vec3(-0.099, 0.147, 0.027);
  s.accel = Vec3(8.148, -0.376, -2.403);
  return s;
}

// The definition of A, checked one error at a time.
TEST(ErrorTransition, PredictsHowEachErrorEvolves) {
  const Filter ours = MakeFilter();
  const ImuSample sample = MakeSample();
  const double dt = 0.005;

  const Mat15 a = ErrorTransition(ours.nav, sample, ours.bias, dt);

  const char* names[] = {"dp", "dv", "dtheta", "dba", "dbg"};
  for (int block = 0; block < 5; ++block) {
    for (int axis = 0; axis < 3; ++axis) {
      Vec15 dx = Vec15::Zero();
      dx(block * 3 + axis) = 1e-6;

      const Filter truth = ApplyError(ours, dx);

      Filter ours_next;
      ours_next.nav = Propagate(ours.nav, sample, ours.bias, dt);
      ours_next.bias = ours.bias;

      Filter truth_next;
      truth_next.nav = Propagate(truth.nav, sample, truth.bias, dt);
      truth_next.bias = truth.bias;

      const Vec15 measured = ErrorBetween(truth_next, ours_next);
      const Vec15 predicted = a * dx;

      EXPECT_LT((measured - predicted).norm(), 1e-11)
          << "error in " << names[block] << ", axis " << axis;
    }
  }
}

TEST(ErrorTransition, IsTheIdentityForAZeroStep) {
  const Filter f = MakeFilter();

  const Mat15 a = ErrorTransition(f.nav, MakeSample(), f.bias, 0.0);

  EXPECT_LT((a - Mat15::Identity()).norm(), 1e-12);
}

// Exp always gives a rotation, and a rotation preserves length.
TEST(ErrorTransition, DoesNotGrowAnOrientationError) {
  const Filter f = MakeFilter();
  const Mat15 a = ErrorTransition(f.nav, MakeSample(), f.bias, 0.005);
  const Mat3 block = a.block<3, 3>(6, 6);

  EXPECT_LT((block.transpose() * block - Mat3::Identity()).norm(), 1e-14);
  EXPECT_NEAR(block.determinant(), 1.0, 1e-14);
}

// The arithmetic done by hand.
TEST(ProcessNoise, MatchesTheHandCalculation) {
  const Mat15 q = ProcessNoise(ImuNoise{}, 0.005);

  EXPECT_NEAR(q(3, 3), 2.000000e-08, 1e-14);
  EXPECT_NEAR(q(6, 6), 1.439565e-10, 1e-16);
  EXPECT_NEAR(q(9, 9), 4.500000e-08, 1e-14);
  EXPECT_NEAR(q(12, 12), 1.880442e-12, 1e-18);
}

// Position gets no fresh noise of its own.
TEST(ProcessNoise, IsDiagonalAndEmptyInThePositionBlock) {
  const Mat15 q = ProcessNoise(ImuNoise{}, 0.005);

  EXPECT_LT((q.block<3, 3>(0, 0).norm()), 1e-20);
  EXPECT_LT((q - Mat15(q.diagonal().asDiagonal())).norm(), 1e-20);
}

// Covariance must stay symmetric, forever.
TEST(CovarianceStep, StaysSymmetric) {
  const Filter f = MakeFilter();
  const ImuSample sample = MakeSample();
  Mat15 p = Mat15::Identity() * 1e-4;

  for (int step = 0; step < 2000; ++step) {
    const Mat15 a = ErrorTransition(f.nav, sample, f.bias, 0.005);
    p = a * p * a.transpose() + ProcessNoise(ImuNoise{}, 0.005);
  }

  EXPECT_LT((p - p.transpose()).norm() / p.norm(), 1e-12);
}

}  // namespace
}  // namespace lvio