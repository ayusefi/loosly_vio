#include <cstdio>
#include <string>
#include <vector>

#include "lvio/imu/propagate.h"
#include "lvio/io/euroc.h"
#include "lvio/io/trajectory.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <mav0 directory> <output.tum>\n", argv[0]);
    return 1;
  }
  const std::string mav0 = argv[1];

  const std::vector<lvio::ImuSample> imu =
      lvio::ReadImu(mav0 + "/imu0/data.csv");
  const std::vector<lvio::GroundTruthSample> truth =
      lvio::ReadGroundTruth(mav0 + "/state_groundtruth_estimate0/data.csv");

  if (imu.empty() || truth.empty()) {
    std::fprintf(stderr, "empty input\n");
    return 1;
  }

  lvio::NavState state;
  state.time = truth.front().time;
  state.pose = truth.front().pose;
  state.velocity = truth.front().velocity;

  lvio::ImuBias bias;
  bias.gyro = truth.front().gyro_bias;
  bias.accel = truth.front().accel_bias;

  std::vector<lvio::TimedValue<lvio::Pose>> output;

  for (std::size_t i = 0; i + 1 < imu.size(); ++i) {
    if (imu[i].time < state.time) {
      continue;  // the ground truth starts later than the imu
    }
    const double dt = lvio::ToSeconds(imu[i + 1].time - imu[i].time);
    state = lvio::Propagate(state, imu[i], bias, dt);
    output.push_back({state.time, state.pose});
  }

  lvio::WriteTumTrajectory(argv[2], output);
  std::printf("propagated %zu samples to %s\n", output.size(), argv[2]);
  return 0;
}