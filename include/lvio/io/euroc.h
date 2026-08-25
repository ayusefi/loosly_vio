#pragma once

#include <string>
#include <vector>

#include "lvio/common/types.h"
#include "lvio/common/time.h"

namespace lvio {

    // One row of a EuRoC data.csv, split into its comma-separated fields.
    using CsvRow = std::vector<std::string>;

    // Reads a EuRoC data.csv. Skips the single comment header, splits each row on
    // commas, and strips the trailing carriage return the files carry. Throws
    // std::runtime_error if the file cannot be opened or a row is ragged.
    std::vector<CsvRow> ReadCsv(const std::string& path);

    // One IMU measurement. Gyroscope and accelerometer, both in the body frame.
    struct ImuSample {
        Timestamp time = 0;
        Vec3 gyro = Vec3::Zero();
        Vec3 accel = Vec3::Zero();
    };

    // Reads mav0/imu0/data.csv.
    std::vector<ImuSample> ReadImu(const std::string& path);
    
} // namespace lvio