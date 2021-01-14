/**
 * @file tracked_bbox.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Tracked bounding box (track)
 * @version 0.1
 * @date 2020-12-22
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "tracked_bbox.hpp"

#include <opencv2/core.hpp>

TrackedBBox::TrackedBBox(const cv::Rect2f& initBbox, float dt)
    : _age(0),
      _numHits(0),
      _numConsecutiveHits(0) {
    // Assign initialize bbox and set velocities to zero
    auto initState = KF::State::all(0.0F);
    auto initMeasurement = rectToMeasurement(initBbox);
    initState(0) = initMeasurement(0);
    initState(1) = initMeasurement(1);
    initState(2) = initMeasurement(2);
    initState(3) = initMeasurement(3);

    // clang-format off

    // Init kalman filter
    _kf.setState(initState);

    // Put high uncertainty on the initial bbox velocities
    _kf.setStateCovMatrix({
        1e1, 0,   0,   0,   0,   0,   0,
        0,   1e1, 0,   0,   0,   0,   0,
        0,   0,   1e1, 0,   0,   0,   0,
        0,   0,   0,   1e1, 0,   0,   0,
        0,   0,   0,   0,   1e4, 0,   0,
        0,   0,   0,   0,   0,   1e4, 0,
        0,   0,   0,   0,   0,   0,   1e4,
    });

    // Set state transition matrix
    _kf.setStateTransitionMatrix({
        1,  0,  0,  0, dt,  0,  0,
        0,  1,  0,  0,  0, dt,  0,
        0,  0,  1,  0,  0,  0, dt,
        0,  0,  0,  1,  0,  0,  0,
        0,  0,  0,  0,  1,  0,  0,
        0,  0,  0,  0,  0,  1,  0,
        0,  0,  0,  0,  0,  0,  1,
    });

    // Set control transition matrix
    _kf.setControlTransitionMatrix({
         0.5F * dt * dt, 0, 
         0,  0.5F * dt * dt,
         0,  0, 
         0,  0, 
        dt,  0, 
         0, dt, 
         0,  0, 
    });

    // Set process noise covariance
    _kf.setProcessNoiseCovMatrix({
        1e0, 0,   0,   0,    0,    0,    0,
        0,   1e0, 0,   0,    0,    0,    0,
        0,   0,   1e0, 0,    0,    0,    0,
        0,   0,   0,   1e-2, 0,    0,    0,
        0,   0,   0,   0,    1e-2, 0,    0,
        0,   0,   0,   0,    0,    1e-2, 0,
        0,   0,   0,   0,    0,    0,    1e-4,
    });
    
    // Set measurement matrix
    _kf.setMeasurementMatrix({
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
    });

    // Set measuremnet noise covariance
    _kf.setMeasurementNoiseCovMatrix({
        1e0, 0,   0,   0,
        0,   1e0, 0,   0,
        0,   0,   1e1, 0,
        0,   0,   0,   1e1, 
    });

    // clang-format on
}

cv::Rect2f TrackedBBox::predict(const cv::Point2f& acceleration) {
    _age++;
    auto statePrior = _kf.predict(acceleration);
    return measurementToRect(statePrior.get_minor<4, 1>(0, 0));
}

cv::Rect2f TrackedBBox::update(const cv::Rect2f& detectedBBox) {
    _numHits++;
    if (_age == 1) {
        _numConsecutiveHits++;
    } else {
        _numConsecutiveHits = 0;
    }
    // Reset age
    _age = 0;
    auto measurement = rectToMeasurement(detectedBBox);
    auto statePosterior = _kf.update(measurement);
    return measurementToRect(statePosterior.get_minor<4, 1>(0, 0));
}

cv::Rect2f TrackedBBox::getRect() const {
    auto state = _kf.getState();
    return measurementToRect(state.get_minor<4, 1>(0, 0));
}

cv::Point2f TrackedBBox::getVelocity() const {
    auto state = _kf.getState();
    auto velocity = state.get_minor<2, 1>(4, 0);
    return {velocity(0), velocity(1)};
};

cv::Matx<float, 4, 1> TrackedBBox::rectToMeasurement(const cv::Rect2f& rect) {
    KF::Measurement measurement;
    float width = rect.width;
    float height = rect.height;
    measurement(0) = rect.x + width * 0.5F;
    measurement(1) = rect.y + height * 0.5F;
    measurement(2) = width * height;
    measurement(3) = width / height;

    return measurement;
}

cv::Rect2f TrackedBBox::measurementToRect(const KF::Measurement& measurement) {
    // Area or aspect ratio is negative, return an empty bbox
    if (measurement(2) < 0.0F || measurement(3) < 0.0F) {
        return {0.0F, 0.0F, 0.0F, 0.0F};
    }
    
    float width = std::sqrt(measurement(2) * measurement(3));
    float height = measurement(2) / width;
    float x = measurement(0) - width * 0.5F;
    float y = measurement(1) - height * 0.5F;

    return {x, y, width, height};
}
