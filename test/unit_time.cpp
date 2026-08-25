#include <gtest/gtest.h>

#include "lvio/common/time.h"
#include "lvio/common/types.h"

namespace lvio {
namespace {

TEST(Time, DurationsSurviveTheRoundTrip) {
  const Timestamp durations[] = {0, 1, 4999936, 5000000, 50000000, 184050000000};

  for (Timestamp d : durations) {
    EXPECT_EQ(FromSeconds(ToSeconds(d)), d) << "for " << d;
  }
}

TEST(Time, HandlesNegativeDurations) {
  EXPECT_EQ(FromSeconds(ToSeconds(-4999936)), -4999936);
}

// The reason Timestamp is an integer: a real dataset timestamp needs 19 digits.
TEST(Time, AbsoluteTimestampsNeedMoreThanADoubleHas) {
  const Timestamp t = 1403636579763555584;
  const Timestamp nearby = t + 100;

  EXPECT_NE(t, nearby);
  EXPECT_EQ(static_cast<double>(t), static_cast<double>(nearby));
}

TEST(TimedValue, DefaultsAreZeroed) {
  const TimedValue<double> a;
  const TimedValue<Vec3> b;

  EXPECT_EQ(a.time, 0);
  EXPECT_DOUBLE_EQ(a.value, 0.0);
  EXPECT_LT(b.value.norm(), 1e-15);
}

}  // namespace
}  // namespace lvio