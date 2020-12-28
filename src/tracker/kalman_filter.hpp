/**
 * @file kalman_filter.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Templated Kalman filter
 * @version 0.1
 * @date 2020-12-15
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <opencv2/core.hpp>

/**
 * @brief Kalman filter
 *
 * @tparam DimState Dimension of state vector
 * @tparam DimMeasurement Dimension of measurement vector
 * @tparam DimControl Dimension of control vector (default is 0)
 */
template <size_t DimState, size_t DimMeasurement, size_t DimControl = 0U>
class KalmanFilter final {
  public:
#pragma region Public types

    using State = cv::Matx<float, DimState, 1>;
    using StateCovMatrix = cv::Matx<float, DimState, DimState>;
    using StateTransitionMatrix = cv::Matx<float, DimState, DimState>;
    using ProcessNoiseCovMatrix = cv::Matx<float, DimState, DimState>;

    using Measurement = cv::Matx<float, DimMeasurement, 1>;
    using MeasurementMatrix = cv::Matx<float, DimMeasurement, DimState>;
    using MeasurementNoiseCovMatrix =
        cv::Matx<float, DimMeasurement, DimMeasurement>;

    using Control = cv::Matx<float, DimControl, 1>;
    using ControlTransitionMatrix = cv::Matx<float, DimState, DimControl>;

#pragma endregion

#pragma region Public methods

    /**
     * @brief Construct a KalmanFilter object
     *
     * @return KalmanFilter object
     */
    KalmanFilter();

    /**
     * @brief Predict prior state estimate
     *
     * @param u Control vector
     * @return  prior state estimate
     */
    State predict(Control u = Control::all(0.0F));

    /**
     * @brief Update predicted state estimate using current measuremnet
     *
     * @param z Measurement
     * @return  posterior state estimate
     */
    State update(Measurement z);

    void setState(State x) { _x = x; }

    State getState() const { return _x; }

    void setStateCovMatrix(StateCovMatrix P) { _P = P; }

    StateCovMatrix getStateCovMatrix() const { return _P; }

    void setStateTransitionMatrix(StateTransitionMatrix F) { _F = F; }

    StateTransitionMatrix getStateTransitionMatrix() const { return _F; }

    void setControlTransitionMatrix(ControlTransitionMatrix B) { _B = B; };

    ControlTransitionMatrix getControlTransitionMatrix() const { return _B; }

    void setProcessNoiseCovMatrix(ProcessNoiseCovMatrix Q) { _Q = Q; }

    ProcessNoiseCovMatrix getProcessNoiseCovMatrix() const { return _Q; }

    void setMeasurementMatrix(MeasurementMatrix H) { _H = H; }

    MeasurementMatrix getMeasurementMatrix() const { return _H; }

    void setMeasurementNoiseCovMatrix(MeasurementNoiseCovMatrix R) { _R = R; }

    MeasurementNoiseCovMatrix getMeasurementNoiseCovMatrix() const {
        return _R;
    }

#pragma endregion

  private:
#pragma region Private member variables

    /**
     * @brief State estimate
     */
    State _x;

    /**
     * @brief State convariance matrix
     */
    StateCovMatrix _P;

    /**
     * @brief Identity matrix with the same size of state covariance matrix
     */
    StateCovMatrix _I;

    /**
     * @brief State transition matrix
     */
    StateTransitionMatrix _F;

    /**
     * @brief Control transition matrix
     */
    ControlTransitionMatrix _B;

    /**
     * @brief Process noise covariance matrix
     */
    ProcessNoiseCovMatrix _Q;

    /**
     * @brief Measurement matrix
     */
    MeasurementMatrix _H;

    /**
     * @brief Measurement noise covariance matrix
     */
    MeasurementNoiseCovMatrix _R;

#pragma endregion
};