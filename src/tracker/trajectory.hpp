/**
 * @file trajectory.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Tracked falling object trajectory
 * @version 0.1
 * @date 2020-12-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Tracked trajectory class
 */
class Trajectory final {
  public:
#pragma region Public types

    using Timestamp = std::chrono::system_clock::time_point;
    using Duration = std::chrono::system_clock::duration;

#pragma endregion

#pragma region Public member methods

    /**
     * @brief Construct a new TrackedTrajectory object
     *
     * @param firstFrame Current frame
     * @return
     */
    Trajectory(const cv::Mat& firstFrame);

    /**
     * @brief Add a bbox sample into this trajectory
     *
     * @param bbox Rect of the added bbox {x, y, w, h}
     * @param velocity Velocity of the added bbox {v_x, v_y}
     * @param timestamp Timestamp of system_clock
     * @return
     */
    void add(const cv::Rect2f& bbox,
             const cv::Point2f& velocity,
             Timestamp timestamp);

    /**
     * @brief Increment the age count of this trajectory
     *
     * @param count Increment count
     * @return
     */
    void incrementAge(int count = 1) { _age += count; }

    /**
     * @brief Get the age count of this trajectory
     *
     * @return  Age count
     */
    int getAge() const { return _age; }

    /**
     * @brief Visualize all samples and fitted trajectory and annotate on the
     * first frame
     *
     * @param anno Output annotations image
     * @return
     */
    void draw(cv::Mat& anno) const;

    /**
     * @brief Get the sample count of this trajectory
     *
     * @return  Number of samples
     */
    size_t getNumSamples() const { return _samples.size(); }

    /**
     * @brief Get the timestamp at the beginning of this trajectory
     * 
     * @return  Start timestamp
     */
    Timestamp getStartTime() const {
        return _samples.empty() ? Timestamp::min() : _samples.front().timestamp;
    }

    /**
     * @brief Get the duration of this trajectory
     *
     * @return  Duration
     */
    Duration getDuration() const {
        return _samples.empty()
                   ? Duration::zero()
                   : _samples.back().timestamp - _samples.front().timestamp;
    }

    /**
     * @brief Get the X range of this trajectory
     *
     * @return  X range
     */
    float getRangeX() const {
        return _samples.empty() ? 0.0F
                                : std::abs(_samples.back().xCenter -
                                           _samples.front().xCenter);
    }

    /**
     * @brief Get the Y range of this trajectory
     *
     * @return  Y range
     */
    float getRangeY() const {
        return _samples.empty() ? 0.0F
                                : std::abs(_samples.back().yCenter -
                                           _samples.front().yCenter);
    }

#pragma endregion

  private:
#pragma region Private types

    /**
     * @brief Sample point class
     */
    struct SamplePoint {
        float x;             // X (left) coordinate of the bbox
        float y;             // Y (top) coordinate of the bbox
        float width;         // Width of the bbox
        float height;        // Height of the bbox
        float xCenter;       // X (center) coordinate of the bbox
        float yCenter;       // Y (center) coordinate of the bbox
        float xVelocity;     // X-direction velocity of the bbox
        float yVelocity;     // Y-direction velocity of the bbox
        Timestamp timestamp; // Timestamp when the bbox is added
    };

#pragma endregion

#pragma region Private constants

    /**
     * @brief Scale factor that maps the velocity to number of pixels in frame
     */
    static constexpr float VELOCITY_SCALE_FACTOR = 0.75F;

    /**
     * @brief X sample step along the fitted parabola
     */
    static constexpr float DRAW_POLYLINE_STEP_X = 0.5F;

#pragma endregion

#pragma region Private member variables

    /**
     * @brief Frame captured when this trajectory started
     */
    cv::Mat _firstFrame;

    /**
     * @brief Sample points added along the trajectory
     */
    std::vector<SamplePoint> _samples;

    /**
     * @brief Age count used to determine whether this trajectory is ended
     */
    int _age;

#pragma endregion

#pragma region Static helper methods

    /**
     * @brief Fit a parabola using sample points
     *
     * @param samples Sample points
     * @param parameters Parabola parameters [a, b, c]^T
     * @return  True: Successfully fitted; False: Cannot fit a parabola
     */
    static bool fitParabola(const std::vector<SamplePoint>& samples,
                            cv::Vec3f& parameters);

#pragma endregion
};