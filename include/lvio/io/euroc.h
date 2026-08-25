#pragma once

#include <string>
#include <vector>
#include <optional>
#include <opencv2/core.hpp>

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

    // A stereo pair with its pixels loaded.
    struct StereoFrame {
        Timestamp time = 0;
        cv::Mat left;
        cv::Mat right;
    };

    // Where the images are, without loading them. All of MH_01 would be 2.7 GB.
    struct StereoStream {
        std::vector<Timestamp> times;
        std::vector<std::string> left_paths;
        std::vector<std::string> right_paths;
    };

    // Reads the cam0 and cam1 index files from a mav0 directory. Throws if the two
    // cameras disagree about any timestamp, which they never do in a healthy
    // sequence but would if an archive were partly extracted.
    StereoStream ReadStereoStream(const std::string& mav0_dir);

    // Loads one pair. Throws if either file is missing or is not 8-bit single
    // channel.
    StereoFrame LoadFrame(const StereoStream& stream, std::size_t index);

    // Calls on_imu and on_frame in time order. on_frame receives an index into the
    // stream rather than pixels, so the caller decides whether to load them.
    template <typename ImuFn, typename FrameFn>
    void Replay(const std::vector<ImuSample>& imu, const StereoStream& frames, ImuFn on_imu, FrameFn on_frame) {
        std::size_t i = 0;
        std::size_t f = 0;

        while (i < imu.size() || f < frames.times.size()) {
            const bool imu_left = i < imu.size();
            const bool frame_left = f < frames.times.size();

            // At equal timestamps the imu goes first: the filter must reach the image's
            // instant before it may use the image.
            const bool imu_next = imu_left && (!frame_left || imu[i].time <= frames.times[f]);

            if (imu_next) {
                on_imu(imu[i]);
                ++i;
            } else {
                on_frame(f);
                ++f;
            }
        }
    }
} // namespace lvio