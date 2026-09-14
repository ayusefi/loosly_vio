#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "lvio/io/euroc.h"
#include "lvio/vision/feature_tracker.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: track_features <mav0 dir> <frames>\n");
    return 1;
  }
  const lvio::StereoStream stream = lvio::ReadStereoStream(argv[1]);
  const std::size_t count = std::min(
      static_cast<std::size_t>(std::atoi(argv[2])), stream.times.size());

  lvio::FeatureTracker tracker(lvio::DetectorConfig(), lvio::TrackerConfig(),
                               300);
  for (std::size_t i = 0; i < count; ++i) {
    const lvio::StereoFrame frame = lvio::LoadFrame(stream, i);
    const std::vector<lvio::FeatureTrack>& tracks = tracker.Update(frame.left);
    if (i % 50 == 0) {
      std::printf("frame %4zu   live tracks %4zu\n", i, tracks.size());
    }
  }

  // The living tracks have not ended yet, so their ages belong in the
  // histogram too -- leaving them out would throw away the longest ones.
  std::vector<int> lengths = tracker.FinishedLengths();
  for (const lvio::FeatureTrack& t : tracker.Tracks()) {
    lengths.push_back(t.age);
  }
  std::sort(lengths.begin(), lengths.end());

  const int low[] = {1, 2, 4, 8, 16, 32, 64, 128};
  const int high[] = {1, 3, 7, 15, 31, 63, 127, 1000000};
  std::printf("\ntrack length histogram (%zu tracks)\n", lengths.size());
  for (int b = 0; b < 8; ++b) {
    const auto in_bucket = [&](int v) { return v >= low[b] && v <= high[b]; };
    const int n = static_cast<int>(
        std::count_if(lengths.begin(), lengths.end(), in_bucket));
    // The last bucket has no upper end, so it is written "128+".
    const std::string label =
        high[b] > 1000 ? std::to_string(low[b]) + "+"
                       : std::to_string(low[b]) + "-" + std::to_string(high[b]);
    std::printf("  %-8s %5d  %5.1f%%  %s\n", label.c_str(), n,
                100.0 * n / static_cast<double>(lengths.size()),
                std::string(static_cast<std::size_t>(
                                60.0 * n / static_cast<double>(lengths.size())),
                            '#')
                    .c_str());
  }
  double mean = 0.0;
  for (int v : lengths) {
    mean += v;
  }
  mean /= static_cast<double>(lengths.size());
  std::printf("\n  mean %.1f frames (%.2f s)  median %d  max %d (%.1f s)\n",
              mean, mean * 0.05, lengths[lengths.size() / 2], lengths.back(),
              lengths.back() * 0.05);
  return 0;
}
