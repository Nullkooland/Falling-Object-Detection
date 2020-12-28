/**
 * @file tracked_bbox.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Tracked bounding box (track)
 * @version 0.1
 * @date 2020-12-19
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include "kalman_filter.hpp"

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Tracked bounding box
 */
class TrackedBBox {
  public:
#pragma region Public member methods

    /**
     * @brief Constructor for tracked bounding box class
     *
     * @param initBbox Initial bbox
     * @param dt Time interval (sec) between two consecutive state update
     * @return
     */
    TrackedBBox(const cv::Rect2f& initBbox, float dt = 1.0F);

    /**
     * @brief Predict next state of this bbox
     *
     * @param acceleration Acceleration {a_x, a_y} (control)
     * @return  Prior estimate of the position of this bbox
     */
    cv::Rect2f predict(const cv::Point2f& acceleration);

    /**
     * @brief Update predicted state of this bbox with measurement
     *
     * @param detectedBBox Detected bbox position (measurement)
     * @return  Posterior estimate of the position of this bbox
     */
    cv::Rect2f update(const cv::Rect2f& detectedBBox);

    /**
     * @brief Extract the rect representation of this bbox
     *
     * @return Rect representation {x, y, w, h}
     */
    cv::Rect2f getRect() const;

    /**
     * @brief Extract the XY velocity of this bbox
     *
     * @return  XY velocity {v_x, v_y}
     */
    cv::Point2f getVelocity() const;

    /**
     * @brief Get the age count of this bbox
     * 
     * @return  Age count
     */
    int getAge() const { return _age; }

    /**
     * @brief Get the hit (update) count of this bbox
     * 
     * @return  Hit count
     */
    int getHitCount() const { return _numHits; }

    /**
     * @brief Get the consecutive hit (update) count of this bbox
     * 
     * @return  Hit streak
     */
    int getHitStreak() const { return _numConsecutiveHits; }

#pragma endregion
  private:
#pragma region Private types

    /**
     * @brief Kalman filter for bbox tracking
     *        State:        [x, y, s, r, v_x, v_y, v_s] \in \R^7
     *        Measurement:  [x, y, s, r]                \in \R^4
     *        Control:      [a_x, a_y]                  \in \R^2
     */
    using KF = KalmanFilter<7, 4, 2>;

#pragma endregion

#pragma region Private member variables

    /**
     * @brief Kalman filter instance
     */
    KF _kf;

    /**
     * @brief Age count used to determine whether this bbox is expired
     */
    int _age;

    /**
     * @brief Number of hits (updates) of this bbox
     */
    int _numHits;

    /**
     * @brief Number of consecutive hits (updates) of this bbox
     */
    int _numConsecutiveHits;

#pragma endregion

    /**
     * @brief Convert rect tuple to measurement vector
     *
     * @param bbox Rect representation of the bbox {x_left, y_top, width,
     * height}
     * @return Measurement vector [x_center, y_center, area, aspect_ratio]^T
     */
    static KF::Measurement rectToMeasurement(const cv::Rect2f& rect);

    /**
     * @brief Convert measurement vector to rect tuple
     *
     * @param measurement Measurement vector [x_center, y_center, area,
     * aspect_ratio]^T
     * @return Rect representation {x_left, y_top, width, height}
     */
    static cv::Rect2f measurementToRect(const KF::Measurement& measurement);
};