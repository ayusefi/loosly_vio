#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "lvio/io/trajectory.h"
#include "lvio/math/so3.h"

namespace lvio {
namespace {

std::string TempPath(const std::string& name) {
  return std::string(testing::TempDir()) + name;
}

TEST(WriteTumTrajectory, WritesEightColumnsPerPose) {
  const std::string path = TempPath("eight.tum");
  std::vector<TimedValue<Pose>> poses = {
      {1000000000, Pose(Mat3::Identity(), Vec3(1.0, 2.0, 3.0))},
      {2000000000, Pose(Exp(Vec3(0.1, 0.2, 0.3)), Vec3(4.0, 5.0, 6.0))}};

  WriteTumTrajectory(path, poses);

  std::ifstream file(path);
  std::string line;
  int lines = 0;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    double value = 0.0;
    int count = 0;
    while (fields >> value) {
      ++count;
    }
    EXPECT_EQ(count, 8) << "line " << lines;
    ++lines;
  }
  EXPECT_EQ(lines, 2);
}

// TUM puts w last. Reading it back with w first would give a different
// rotation, silently.
TEST(WriteTumTrajectory, PutsTheScalarPartLast) {
  const std::string path = TempPath("order.tum");
  // A 90 degree turn about z, whose quaternion is (w, x, y, z) =
  // (0.7071, 0, 0, 0.7071). If w went first, column 5 would be 0.7071 and
  // columns 6 and 7 would be zero.
  const Pose pose(Exp(Vec3(0.0, 0.0, M_PI / 2.0)), Vec3::Zero());
  WriteTumTrajectory(path, {{0, pose}});

  std::ifstream file(path);
  double time = 0.0;
  double tx = 0.0, ty = 0.0, tz = 0.0;
  double qx = 0.0, qy = 0.0, qz = 0.0, qw = 0.0;
  file >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

  EXPECT_NEAR(qx, 0.0, 1e-6);
  EXPECT_NEAR(qy, 0.0, 1e-6);
  EXPECT_NEAR(qz, 0.70710678, 1e-6);
  EXPECT_NEAR(qw, 0.70710678, 1e-6);
}

TEST(WriteTumTrajectory, KeepsTheTimestampToTheNanosecond) {
  const std::string path = TempPath("time.tum");
  const Timestamp original = 1403636580838555648;
  WriteTumTrajectory(path, {{original, Pose()}});

  std::ifstream file(path);
  double seconds = 0.0;
  file >> seconds;

  const Timestamp recovered = static_cast<Timestamp>(seconds * 1e9);
  // A double resolves about 238 ns here. Anything within a
  // microsecond is far inside evo's matching tolerance.
  EXPECT_LT(std::abs(recovered - original), 1000);
}

TEST(WriteTumTrajectory, ThrowsWhenThePathIsUnwritable) {
  EXPECT_THROW(WriteTumTrajectory("/no/such/directory/out.tum", {}),
               std::runtime_error);
}

}  // namespace
}  // namespace lvio