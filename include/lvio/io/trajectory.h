#pragma once

#include <string>
#include <vector>

#include "lvio/common/time.h"
#include "lvio/math/pose.h"

namespace lvio {

// Writes poses in TUM format, which is what evo reads:
//
//     timestamp tx ty tz qx qy qz qw
//
// The timestamp is in seconds and the quaternion has w LAST, both of which
// differ from how EuRoC stores the same information. Throws if the file cannot
// be opened.

void WriteTumTrajectory(const std::string& path,
                        const std::vector<TimedValue<Pose>> trajectory);

}  // namespace lvio