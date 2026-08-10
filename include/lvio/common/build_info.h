#pragma once

#include <string>

namespace lvio {

// Version, compiler and build type of the running binary. Every trajectory and
// results file we emit records this, so a number can always be traced back to
// the build that produced it.
std::string BuildSummary();

}  // namespace lvio
