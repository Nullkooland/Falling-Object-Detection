/**
 * @file kalman_filter.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Templated Kalman filter
 * @version 0.1
 * @date 2020-12-15
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "kalman_filter.hpp"

#include <cstddef>

template <size_t DimState, size_t DimMeasurement, size_t DimControl>
KalmanFilter<DimState, DimMeasurement, DimControl>::KalmanFilter() {
    // Initialize internal states
    _x = State::all(0.0F);
    _P = StateCovMatrix::eye();
    _I = StateCovMatrix::eye();
    _F = StateTransitionMatrix::eye();
    _Q = ProcessNoiseCovMatrix::eye();
    _H = MeasurementMatrix::zeros();
    _R = MeasurementNoiseCovMatrix::eye();
}

template <size_t DimState, size_t DimMeasurement, size_t DimControl>
cv::Matx<float, DimState, 1>
KalmanFilter<DimState, DimMeasurement, DimControl>::predict(Control u) {
    // Predict prior state estimate
    if constexpr (DimControl == 0) {
        _x = _F * _x;
    } else {
        _x = _F * _x + _B * u;
    }

    // Predict prior state convariance matrix
    _P = _F * _P * _F.t() + _Q;

    return _x;
}

template <size_t DimState, size_t DimMeasurement, size_t DimControl>
cv::Matx<float, DimState, 1>
KalmanFilter<DimState, DimMeasurement, DimControl>::update(Measurement z) {
    // Calculate Kalman gain
    auto K = _P * _H.t() * (_H * _P * _H.t() + _R).inv();
    // Update posterior state estimate
    _x = _x + K * (z - _H * _x);
    // Update posterior state covariance matrix
    _P = (_I - K * _H) * _P;

    return _x;
}

// Need this disgusting workaround to avoid link error
#include "kalman_filter_instantiation.hpp"