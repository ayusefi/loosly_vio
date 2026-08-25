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

}  // namespace
}  // namespace lvio