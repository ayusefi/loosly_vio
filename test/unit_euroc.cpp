#include "lvio/io/euroc.h"

#include <gtest/gtest.h>

#include <fstream>

namespace lvio {
namespace {

// Writes a file and returns its path, so the parsing tests need no dataset.
std::string WriteTempFile(const std::string& name, const std::string& contents) {
  const std::string path = std::string(testing::TempDir()) + name;
  std::ofstream file(path, std::ios::binary);
  file << contents;
  return path;
}

TEST(ReadCsv, SkipsTheHeaderAndSplitsFields) {
  const std::string path = WriteTempFile("plain.csv",
                                         "#a,b,c\n"
                                         "1,2,3\n"
                                         "4,5,6\n");

  const std::vector<CsvRow> rows = ReadCsv(path);

  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0][0], "1");
  EXPECT_EQ(rows[0][2], "3");
  EXPECT_EQ(rows[1][1], "5");
}

// The defect from part 4. Without the strip, the last field keeps a '\r'.
TEST(ReadCsv, StripsTheCarriageReturn) {
  const std::string path = WriteTempFile("crlf.csv",
                                         "#t,name\r\n"
                                         "100,image.png\r\n");

  const std::vector<CsvRow> rows = ReadCsv(path);

  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0][1], "image.png");
  EXPECT_EQ(rows[0][1].size(), 9u);
}

TEST(ReadCsv, ThrowsWhenTheFileIsMissing) {
  EXPECT_THROW(ReadCsv("/no/such/file.csv"), std::runtime_error);
}

TEST(ReadCsv, ThrowsOnARaggedRow) {
  const std::string path = WriteTempFile("ragged.csv",
                                         "#a,b,c\n"
                                         "1,2,3\n"
                                         "4,5\n");

  EXPECT_THROW(ReadCsv(path), std::runtime_error);
}

TEST(ReadImu, ParsesGyroBeforeAccel) {
  const std::string path =
      WriteTempFile("imu.csv",
                    "#timestamp,wx,wy,wz,ax,ay,az\n"
                    "1403636579758555392,-0.0991347,0.14730578,0.02722713,"
                    "8.14769170,-0.37592158,-2.40262924\n");

  const std::vector<ImuSample> samples = ReadImu(path);

  ASSERT_EQ(samples.size(), 1u);
  EXPECT_EQ(samples[0].time, 1403636579758555392);
  EXPECT_NEAR(samples[0].gyro.x(), -0.0991347, 1e-9);
  EXPECT_NEAR(samples[0].accel.x(), 8.14769170, 1e-9);
  // Gravity dominates the accelerometer; the gyro is small. If these are
  // swapped, this is what notices.
  EXPECT_GT(samples[0].accel.norm(), 5.0);
  EXPECT_LT(samples[0].gyro.norm(), 1.0);
}

// The real thing. Skips rather than fails if the dataset is not present.
TEST(ReadImu, MatchesTheMeasuredFactsForMh01) {
  const std::string path =
      "data/euroc/machine_hall/MH_01_easy/mav0/imu0/data.csv";
  std::ifstream exists(path);
  if (!exists) {
    GTEST_SKIP() << "dataset not present at " << path;
  }

  const std::vector<ImuSample> samples = ReadImu(path);

  EXPECT_EQ(samples.size(), 36820u);

  for (size_t i = 1; i < samples.size(); ++i) {
    ASSERT_GT(samples[i].time, samples[i - 1].time) << "at row " << i;
  }

  const Timestamp span = samples.back().time - samples.front().time;
  const double rate = static_cast<double>(samples.size() - 1) / ToSeconds(span);
  EXPECT_NEAR(rate, 200.0, 0.1);
}

// Builds a short synthetic ground truth: a pure turn about z at 100 Hz.
std::vector<GroundTruthSample> MakeSamples() {
  std::vector<GroundTruthSample> samples;
  for (int i = 0; i < 5; ++i) {
    GroundTruthSample s;
    s.time = 1000000000 + i * 10000000;  // 10 ms apart
    s.pose = Pose(Exp(Vec3(0.0, 0.0, 0.1 * i)), Vec3(i, 2 * i, 0.0));
    samples.push_back(s);
  }
  return samples;
}

TEST(PoseAt, ReturnsNothingBeforeTheFirstSample) {
  const auto samples = MakeSamples();

  EXPECT_FALSE(PoseAt(samples, samples.front().time - 1).has_value());
}

TEST(PoseAt, ReturnsNothingAfterTheLastSample) {
  const auto samples = MakeSamples();

  EXPECT_FALSE(PoseAt(samples, samples.back().time + 1).has_value());
}

TEST(PoseAt, ReturnsNothingForAnEmptyList) {
  EXPECT_FALSE(PoseAt({}, 1000000000).has_value());
}

TEST(PoseAt, ReproducesTheSamplesExactly) {
  const auto samples = MakeSamples();

  for (const GroundTruthSample& s : samples) {
    const auto found = PoseAt(samples, s.time);

    ASSERT_TRUE(found.has_value()) << "at " << s.time;
    EXPECT_LT((found->R - s.pose.R).norm(), 1e-15);
    EXPECT_LT((found->t - s.pose.t).norm(), 1e-15);
  }
}

TEST(PoseAt, InterpolatesHalfwayBetweenTwoSamples) {
  const auto samples = MakeSamples();
  const Timestamp mid = (samples[1].time + samples[2].time) / 2;

  const auto found = PoseAt(samples, mid);

  ASSERT_TRUE(found.has_value());
  EXPECT_LT((found->t - 0.5 * (samples[1].pose.t + samples[2].pose.t)).norm(),
            1e-12);
  const double covered = Log(samples[1].pose.R.transpose() * found->R).norm();
  const double total =
      Log(samples[1].pose.R.transpose() * samples[2].pose.R).norm();
  EXPECT_NEAR(covered, 0.5 * total, 1e-12);
}

TEST(ReadGroundTruth, NormalisesTheQuaternions) {
  const std::string path =
      "data/euroc/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/"
      "data.csv";
  std::ifstream exists(path);
  if (!exists) {
    GTEST_SKIP() << "dataset not present";
  }

  const std::vector<GroundTruthSample> samples = ReadGroundTruth(path);

  ASSERT_EQ(samples.size(), 36382u);
  for (const GroundTruthSample& s : samples) {
    ASSERT_LT((s.pose.R.transpose() * s.pose.R - Mat3::Identity()).norm(), 1e-12)
        << "at " << s.time;
  }
}

// The ground truth does not cover the whole sequence.
TEST(PoseAt, RefusesTheFramesMh01DoesNotCover) {
  const std::string dir = "data/euroc/machine_hall/MH_01_easy/mav0/";
  std::ifstream exists(dir + "imu0/data.csv");
  if (!exists) {
    GTEST_SKIP() << "dataset not present";
  }

  const auto gt = ReadGroundTruth(dir + "state_groundtruth_estimate0/data.csv");
  const std::vector<CsvRow> images = ReadCsv(dir + "cam0/data.csv");

  int covered = 0;
  int refused = 0;
  for (const CsvRow& row : images) {
    if (PoseAt(gt, std::stoll(row[0])).has_value()) {
      ++covered;
    } else {
      ++refused;
    }
  }

  EXPECT_EQ(covered, 3638);
  EXPECT_EQ(refused, 44);
}

}  // namespace
}  // namespace lvio