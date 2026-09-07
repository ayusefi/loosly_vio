#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "lvio/imu/propagate.h"
#include "lvio/io/euroc.h"
#include "lvio/io/trajectory.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s <mav0 directory> <output.tum> <drift.csv>"
                 " <initial bias sigma>\n",
                 argv[0]);
    return 1;
  }
  const std::string mav0 = argv[1];
  const double bias_sigma = std::atof(argv[4]);

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

  lvio::Mat15 covariance = lvio::Mat15::Zero();

  // Claiming we know the biases exactly is what made the
  // envelope dishonest. A non-zero sigma here admits we do not.
  covariance.block<3, 3>(9, 9) =
      lvio::Mat3::Identity() * bias_sigma * bias_sigma;
  covariance.block<3, 3>(12, 12) =
      lvio::Mat3::Identity() * (bias_sigma * 0.03) * (bias_sigma * 0.03);

  const lvio::ImuNoise noise;

  std::ofstream drift(argv[3]);
  if (!drift) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  drift << "#seconds,drift_m,three_sigma_m\n";

  std::vector<lvio::TimedValue<lvio::Pose>> output;

  for (std::size_t i = 0; i + 1 < imu.size(); ++i) {
    if (imu[i].time < state.time) {
      continue;  // the ground truth starts later than the imu
    }
    const double dt = lvio::ToSeconds(imu[i + 1].time - imu[i].time);
    state = lvio::Propagate(state, imu[i], bias, dt);
    output.push_back({state.time, state.pose});

    const lvio::Mat15 a = lvio::ErrorTransition(state, imu[i], bias, dt);
    covariance = a * covariance * a.transpose() + lvio::ProcessNoise(noise, dt);

    const std::optional<lvio::Pose> reference = lvio::PoseAt(truth, state.time);
    if (reference.has_value()) {
      const double actual = (state.pose.t - reference->t).norm();
      // The three position variances sit on the diagonal of the dp block.
      const double three_sigma =
          3 * std::sqrt(covariance.block<3, 3>(0, 0).trace() / 3);

      drift << lvio::ToSeconds(state.time - truth.front().time) << ',' << actual
            << ',' << three_sigma << '\n';
    }
  }

  lvio::WriteTumTrajectory(argv[2], output);
  std::printf("propagated %zu samples to %s\n", output.size(), argv[2]);
  return 0;
}