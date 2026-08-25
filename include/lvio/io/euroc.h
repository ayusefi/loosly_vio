#pragma once

#include <string>
#include <vector>
#include <optional>

#include "lvio/common/types.h"
#include "lvio/common/time.h"
#include "lvio/math/pose.h"

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

    // One row of the ground truth. The pose is T_WB: body frame into world frame.
    // Velocity is in world axes, both biases in body axes.
    struct GroundTruthSample {
        Timestamp time = 0;
        Pose pose;
        Vec3 velocity = Vec3::Zero();
        Vec3 gyro_bias = Vec3::Zero();
        Vec3 accel_bias = Vec3::Zero();
    };

    // Reads mav0/state_groundtruth_estimate0/data.csv. The stored quaternions are
    // rounded to six decimals and are not quite unit length, so they are normalised
    // here rather than anywhere downstream.
    std::vector<GroundTruthSample> ReadGroundTruth(const std::string& path);

    // The pose at an arbitrary time, interpolated between the two rows either side.
    // Returns nothing if t falls outside the covered interval -- 44 images of MH_01
    // do. Samples must be sorted by time.
    std::optional<Pose> PoseAt(const std::vector<GroundTruthSample>& samples, Timestamp t);
    
} // namespace lvio