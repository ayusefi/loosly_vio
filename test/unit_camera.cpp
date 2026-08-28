#include <gtest/gtest.h>

#include <fstream>

#include "lvio/camera/camera.h"

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

// Nothing to undo at the centre.
TEST(Undistort, DoesNothingAtTheCentre) {
  const Camera c = MakeCam0();

  EXPECT_LT(c.Undistort(Vec2::Zero()).norm(), 1e-15);
}

// The test that ties the two directions together. Whatever Distort does,
// Undistort must undo, everywhere in the image including the corners.
TEST(Undistort, UndoesDistortAcrossTheWholeImage) {
  const Camera c = MakeCam0();

  for (int v = 0; v <= c.height; v += 60) {
    for (int u = 0; u <= c.width; u += 94) {
      const Vec2 original((u - c.cu) / c.fu, (v - c.cv) / c.fv);
      const Vec2 recovered = c.Undistort(c.Distort(original));

      const double error_px = std::hypot((recovered.x() - original.x()) * c.fu,
                                         (recovered.y() - original.y()) * c.fv);
      EXPECT_LT(error_px, 1e-4) << "at pixel " << u << ", " << v;
    }
  }
}

// The full round trip: a 3D point out to a pixel and back to a ray. The ray
// must point at the original.
TEST(Unproject, GivesBackTheDirectionProjectSentIn) {
  const Camera c = MakeCam0();
  const Vec3 points[] = {Vec3(0.0, 0.0, 4.0), Vec3(1.0, 0.5, 5.0),
                         Vec3(-2.0, 1.5, 3.0), Vec3(0.2, -0.9, 1.5)};

  for (const Vec3& point : points) {
    const Vec3 ray = c.Unproject(c.Project(point));

    EXPECT_NEAR(ray.norm(), 1.0, 1e-12) << "ray must be unit length";
    // Same direction: the angle between them is zero, so the cross product is.
    EXPECT_LT(ray.cross(point.normalized()).norm(), 1e-8)
        << "for point " << point.transpose();
  }
}

TEST(Unproject, PointsStraightAheadFromThePrincipalPoint) {
  const Camera c = MakeCam0();

  const Vec3 ray = c.Unproject(Vec2(c.cu, c.cv));

  EXPECT_LT((ray - Vec3(0.0, 0.0, 1.0)).norm(), 1e-12);
}

// The corner is the slow case, and it must still converge.
TEST(Undistort, HandlesTheWorstCorner) {
  const Camera c = MakeCam0();
  const Vec2 corner((0.0 - c.cu) / c.fu, (0.0 - c.cv) / c.fv);

  const Vec2 recovered = c.Undistort(c.Distort(corner));

  EXPECT_LT((recovered - corner).norm(), 1e-7);
}

}  // namespace
}  // namespace lvio
