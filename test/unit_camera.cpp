#include "lvio/camera/camera.h"

#include <gtest/gtest.h>

#include <fstream>

namespace lvio {
namespace {

// The cam0 numbers, so the parsing tests and the maths tests are independent.
Camera MakeCam0() {
  Camera c;
  c.width = 752;
  c.height = 480;
  c.fu = 458.654;
  c.fv = 457.296;
  c.cu = 367.215;
  c.cv = 248.375;
  c.k1 = -0.28340811;
  c.k2 = 0.07395907;
  c.p1 = 0.00019359;
  c.p2 = 1.76187114e-05;
  return c;
}

// At the principal point r2 is 0, so scale is exactly 1.
TEST(Distort, DoesNothingAtTheCentre) {
  const Camera c = MakeCam0();

  const Vec2 out = c.Distort(Vec2::Zero());

  EXPECT_LT(out.norm(), 1e-15);
}

// 88.88 px radial plus 0.17 px of twist gives 89.04 px in total.
TEST(Distort, MovesTheTopLeftCornerBy89Pixels) {
  const Camera c = MakeCam0();

  const Vec2 before((0.0 - c.cu) / c.fu, (0.0 - c.cv) / c.fv);
  const Vec2 after = c.Distort(before);

  const double moved_u = (after.x() - before.x()) * c.fu;
  const double moved_v = (after.y() - before.y()) * c.fv;

  EXPECT_NEAR(std::hypot(moved_u, moved_v), 89.04, 0.01);
}

// The worked example.
TEST(Project, MatchesTheWorkedExample) {
  Camera c = MakeCam0();
  c.k1 = c.k2 = c.p1 = c.p2 = 0.0;  // the example ignored distortion

  const Vec2 pixel = c.Project(Vec3(1.0, 0.0, 5.0));

  EXPECT_NEAR(pixel.x(), 458.9458, 1e-3);
  EXPECT_NEAR(pixel.y(), 248.3750, 1e-3);
}

// Dividing by depth destroys it.
TEST(Project, GivesTheSamePixelForBothPointsOnARay) {
  const Camera c = MakeCam0();

  const Vec2 near = c.Project(Vec3(1.0, 0.0, 5.0));
  const Vec2 far = c.Project(Vec3(2.0, 0.0, 10.0));

  EXPECT_LT((near - far).norm(), 1e-12);
}

TEST(Project, PutsAPointStraightAheadAtThePrincipalPoint) {
  const Camera c = MakeCam0();

  const Vec2 pixel = c.Project(Vec3(0.0, 0.0, 3.0));

  EXPECT_NEAR(pixel.x(), c.cu, 1e-12);
  EXPECT_NEAR(pixel.y(), c.cv, 1e-12);
}

TEST(Project, RefusesPointsAtOrBehindTheCamera) {
  const Camera c = MakeCam0();

  EXPECT_THROW(c.Project(Vec3(1.0, 0.0, 0.0)), std::runtime_error);
  EXPECT_THROW(c.Project(Vec3(1.0, 0.0, -2.0)), std::runtime_error);
}

TEST(ReadCamera, ReadsCam0) {
  const std::string path =
      "data/euroc/machine_hall/MH_01_easy/mav0/cam0/sensor.yaml";
  std::ifstream exists(path);
  if (!exists) {
    GTEST_SKIP() << "dataset not present";
  }

  const Camera c = ReadCamera(path);

  EXPECT_EQ(c.width, 752);
  EXPECT_EQ(c.height, 480);
  EXPECT_DOUBLE_EQ(c.fu, 458.654);
  EXPECT_DOUBLE_EQ(c.cv, 248.375);
  EXPECT_DOUBLE_EQ(c.k1, -0.28340811);
  EXPECT_DOUBLE_EQ(c.p2, 1.76187114e-05);
}

TEST(ReadCamera, ThrowsOnAMissingFile) {
  EXPECT_THROW(ReadCamera("/no/such/sensor.yaml"), std::exception);
}

}  // namespace
}  // namespace lvio
