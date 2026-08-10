#include "lvio/common/build_info.h"

#include <sstream>

#include "lvio/common/version.h"

#ifndef LVIO_BUILD_TYPE
#define LVIO_BUILD_TYPE "unknown"
#endif

namespace lvio {

std::string BuildSummary() {
  std::ostringstream out;
  out << "lvio " << kVersion << " (" << LVIO_BUILD_TYPE << ", gcc " << __VERSION__
      << ")";
  return out.str();
}

}  // namespace lvio
