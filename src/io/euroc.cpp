#include "lvio/io/euroc.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

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
    
} // namespace lvio