#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lvio/io/euroc.h"
#include "lvio/vision/stereo_matcher.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: stereo_depth <mav0 dir> <frames> <out csv> "
                 "[overlay.png frame]\n");
    return 1;
  }
  const lvio::StereoCalibration calibration =
      lvio::ReadStereoCalibration(argv[1]);
  const lvio::StereoStream stream = lvio::ReadStereoStream(argv[1]);
  const std::size_t count = std::min(
      static_cast<std::size_t>(std::atoi(argv[2])), stream.times.size());

  const lvio::StereoMatcherConfig config;
  lvio::FeatureTracker tracker(lvio::DetectorConfig(), lvio::TrackerConfig(),
                               300);

  // Optional: draw one frame's points coloured by depth. Looking at this is
  // the only check that can tell a wrong depth from an unexpected room.
  const std::string overlay = argc > 4 ? argv[4] : std::string();
  const std::size_t overlay_frame =
      argc > 5 ? static_cast<std::size_t>(std::atoi(argv[5])) : 0;

  std::vector<double> depths;
  std::size_t offered = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const lvio::StereoFrame frame = lvio::LoadFrame(stream, i);
    const std::vector<lvio::FeatureTrack>& tracks = tracker.Update(frame.left);
    const std::vector<lvio::StereoPoint> points =
        lvio::MatchStereo(frame.left, frame.right, tracks, calibration, config);
    offered += tracks.size();
    for (const lvio::StereoPoint& p : points) {
      depths.push_back(p.point.z());
    }

    if (!overlay.empty() && i == overlay_frame) {
      cv::Mat picture;
      cv::cvtColor(frame.left, picture, cv::COLOR_GRAY2BGR);
      for (const lvio::StereoPoint& p : points) {
        const double z = p.point.z();
        const cv::Scalar colour = z < 1.0   ? cv::Scalar(0, 0, 255)
                                  : z < 6.0 ? cv::Scalar(0, 255, 255)
                                            : cv::Scalar(0, 255, 0);
        cv::circle(picture, p.left, 4, colour, -1);
      }
      cv::putText(picture, "red <1m  yellow 1-6m  green >6m", cv::Point(10, 24),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
      cv::imwrite(overlay, picture);
      std::printf("wrote %s (frame %zu)\n", overlay.c_str(), i);
    }
  }

  if (depths.empty()) {
    std::fprintf(stderr, "no points survived the gates\n");
    return 1;
  }
  std::sort(depths.begin(), depths.end());
  const auto at = [&](double q) {
    return depths[std::min(
        depths.size() - 1,
        static_cast<std::size_t>(q * static_cast<double>(depths.size())))];
  };

  const double fb = calibration.left.fu * calibration.T_left_right.t.norm();
  const double sigma_d = 0.3;

  std::printf("frames %zu   tracks offered %zu   points kept %zu  (%.1f%%)\n",
              count, offered, depths.size(),
              100.0 * static_cast<double>(depths.size()) /
                  static_cast<double>(offered));
  std::printf("baseline %.6f m   f*b %.3f px*m\n\n",
              calibration.T_left_right.t.norm(), fb);
  std::printf("depth percentiles:\n");
  for (const double q : {0.01, 0.25, 0.50, 0.75, 0.95, 0.99}) {
    std::printf("   p%-3.0f  %8.2f m\n", q * 100.0, at(q));
  }
  std::printf(
      "\nuncertainty from sigma_z = z^2 * sigma_d / (f*b), sigma_d = %.1f px:\n",
      sigma_d);
  std::printf("     depth    sigma_z   relative\n");
  for (const double z : {1.0, 2.0, 5.0, 10.0, 15.0, 17.0, 20.0, 25.0}) {
    const double s = z * z * sigma_d / fb;
    std::printf("   %6.1f m  %8.4f m   %6.2f%%%s\n", z, s, 100.0 * s / z,
                z > config.max_depth ? "   <- gated out" : "");
  }
  std::printf("\n10%% relative uncertainty is reached at z = %.2f m; "
              "max_depth is %.1f m\n",
              0.10 * fb / sigma_d, config.max_depth);

  std::ofstream csv(argv[3]);
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  csv << "#depth_m,sigma_z_m,relative\n";
  for (const double z : depths) {
    const double s = z * z * sigma_d / fb;
    csv << z << ',' << s << ',' << s / z << '\n';
  }
  std::printf("wrote %s\n", argv[3]);
  return 0;
}