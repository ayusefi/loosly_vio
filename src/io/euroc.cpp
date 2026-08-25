#include "lvio/io/euroc.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>

namespace lvio {
    namespace {
        // getline removes the newline but leaves the carriage return that three of the
        // four EuRoC csv files carry. Numbers survive it; filenames do not.  
        std::string StripCarriageReturn(const std::string& line) {
            if (!line.empty() && line.back() == '\r') {
                return line.substr(0, line.size() - 1);
            }
            return line;
        }
    } // namespace

    std::vector<CsvRow> ReadCsv(const std::string& path) {
        std::ifstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open " + path);
        }

        std::vector<CsvRow> rows;
        std::string line;
        bool first = true;

        while (std::getline(file, line)) {

            line = StripCarriageReturn(line);

            if (first) {
                first = false;
                continue; // the comment header
            }
            if (line.empty()) {
                continue;
            }

            std::stringstream fields(line);
            CsvRow row;
            std::string field;
            while (std::getline(fields, field, ',')) {
                row.push_back(field);
            }

            if (!rows.empty() && row.size() != rows.front().size()) {
                throw std::runtime_error("Ragged row in " + path);
            }
            rows.push_back(row);
        }
        return rows;
    }

    std::vector<ImuSample> ReadImu(const std::string& path) {
        const std::vector<CsvRow> rows = ReadCsv(path);
        std::vector<ImuSample> samples;
        samples.reserve(rows.size());

        for (const CsvRow& row : rows) {
            if (row.size() != 7) {
                throw std::runtime_error("Expected 7 fields in " + path);
            }
            ImuSample sample;
            sample.time = std::stoll(row[0]);
            sample.gyro.x() = std::stod(row[1]);
            sample.gyro.y() = std::stod(row[2]);
            sample.gyro.z() = std::stod(row[3]);
            sample.accel.x() = std::stod(row[4]);
            sample.accel.y() = std::stod(row[5]);
            sample.accel.z() = std::stod(row[6]);
            samples.push_back(sample);
        }
        return samples;
    }

    std::vector<GroundTruthSample> ReadGroundTruth(const std::string& path) {
        const std::vector<CsvRow> rows = ReadCsv(path);

        std::vector<GroundTruthSample> samples;
        samples.reserve(rows.size());

        for (const CsvRow& row : rows) {
            if (row.size() != 17) {
                throw std::runtime_error("Expected 17 fields in " + path);
            }

            const Quat quat(std::stod(row[4]), std::stod(row[5]),
                            std::stod(row[6]), std::stod(row[7]));

            GroundTruthSample sample;
            sample.time = std::stoll(row[0]);
            // normalize the quaternion because the stored values are rounded to six decimals
            sample.pose = Pose(quat.normalized().toRotationMatrix(),
                               Vec3(std::stod(row[1]), std::stod(row[2]),
                                    std::stod(row[3])));
            sample.velocity = Vec3(std::stod(row[8]), std::stod(row[9]), std::stod(row[10]));
            sample.gyro_bias = Vec3(std::stod(row[11]), std::stod(row[12]), std::stod(row[13]));
            sample.accel_bias = Vec3(std::stod(row[14]), std::stod(row[15]), std::stod(row[16]));
            samples.push_back(sample);
        }
        return samples;
    }

    std::optional<Pose> PoseAt(const std::vector<GroundTruthSample>& samples, Timestamp t) {
        if (samples.empty() || t < samples.front().time || t > samples.back().time) {
            return std::nullopt;
        }

        const auto it = std::lower_bound(samples.begin(), samples.end(), t,
            [](const GroundTruthSample& sample, Timestamp time) {
                return sample.time < time;
            });
        
        if (it->time == t) {
            return it->pose;
        }

        const GroundTruthSample& after = *it;
        const GroundTruthSample& before = *(it - 1);

        const double s = static_cast<double>(t - before.time) / 
                        (static_cast<double>(after.time - before.time));
        return Interpolate(before.pose, after.pose, s);
    }

    StereoStream ReadStereoStream(const std::string& mav0_dir) {
        const std::vector<CsvRow> left = ReadCsv(mav0_dir + "/cam0/data.csv");
        const std::vector<CsvRow> right = ReadCsv(mav0_dir + "/cam1/data.csv");

        if (left.size() != right.size()) {
            throw std::runtime_error("cam0 and cam1 have different lengths in " + mav0_dir);
        }

        StereoStream stream;
        stream.times.reserve(left.size());
        stream.left_paths.reserve(left.size());
        stream.right_paths.reserve(right.size());

        for (std::size_t i = 0; i < left.size(); ++i) {
            if (left[i].front() != right[i].front()) {
                throw std::runtime_error("cam0 and cam2 timestamps differ at " + std::to_string(i));
            }

            stream.times.push_back(std::stoll(left[i][0]));
            stream.left_paths.push_back(mav0_dir + "/cam0/data/" + left[i][1]);
            stream.right_paths.push_back(mav0_dir + "/cam1/data/" + right[i][1]);
        }

        return stream;
    }

    StereoFrame LoadFrame(const StereoStream& stream, std::size_t index) {
        if (index >= stream.times.size()) {
            throw std::runtime_error("frame index out of range");
        }

        StereoFrame frame;
        frame.time = stream.times[index];
        frame.left = cv::imread(stream.left_paths[index], cv::IMREAD_GRAYSCALE);
        frame.right = cv::imread(stream.right_paths[index], cv::IMREAD_GRAYSCALE);

        for (const cv::Mat& image : {frame.left, frame.right}) {
            if (image.empty()) {
                throw std::runtime_error("cannot read image for frame " + std::to_string(index));
            }

            if (image.type() != CV_8UC1) {
                throw std::runtime_error("expected 8-bit single channel");
            }
        }

        return frame;
    }


} // namespace lvio