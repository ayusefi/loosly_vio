#include <cstdio>
#include <string>
#include <vector>

#include "lvio/io/euroc.h"
#include "lvio/io/trajectory.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s euroc_dir tum_file\n", argv[0]);
    return 1;
  }

  const std::string mav0 = argv[1];
  const std::string output = argv[2];

  const std::vector<lvio::GroundTruthSample> samples =
      lvio::ReadGroundTruth(mav0 + "/state_groundtruth_estimate0/data.csv");

  std::vector<lvio::TimedValue<lvio::Pose>> poses;
  poses.reserve(samples.size());
  for (const lvio::GroundTruthSample& sample : samples) {
    poses.push_back({sample.time, sample.pose});
  }

  lvio::WriteTumTrajectory(output, poses);
  std::printf("Wrote %zu poses to %s\n", poses.size(), output.c_str());

  return 0;
}