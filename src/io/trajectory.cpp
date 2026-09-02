#include "lvio/io/trajectory.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace lvio {

void WriteTumTrajectory(const std::string& path,
                        const std::vector<TimedValue<Pose>> poses) {
  std::ofstream file(path);

  if (!file) {
    throw std::runtime_error("File not found! Could not open " + path);
  }

  file << std::fixed;

  for (const TimedValue<Pose>& entry : poses) {
    // TUM wants absolute seconds.
    const double seconds = static_cast<double>(entry.time) / 1e9;

    const Quat q(entry.value.R);
    const Vec3 t = entry.value.t;

    file << std::setprecision(9) << seconds << " " << std::setprecision(6)
         << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
         << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
}
}  // namespace lvio