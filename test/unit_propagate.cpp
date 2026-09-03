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

}  // namespace
}  // namespace lvio