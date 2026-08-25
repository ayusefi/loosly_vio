#pragma once

#include <cstdint>


namespace lvio {
    using Timestamp = std::int64_t;

    constexpr double ToSeconds(Timestamp duration) {
        return static_cast<double>(duration) * 1e-9;
    }

    constexpr Timestamp FromSeconds(double duration) {
        return static_cast<Timestamp>(duration * 1e9 + (duration < 0 ? -0.5 : 0.5));
    }

    template <typename T>
    struct TimedValue {
        Timestamp time = 0;
        T value{};
    };
    
} // namespace lvio