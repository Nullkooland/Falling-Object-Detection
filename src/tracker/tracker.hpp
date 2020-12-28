/**
 * @file tracker.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Falling object bbox tracker based on SORT algorithm
 * @version 0.1
 * @date 2020-12-27
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include "lap_solver.hpp"
#include "tracked_bbox.hpp"
#include "trajectory.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>

/**
 * @brief Falling object bbox tracker class
 */
class SortTracker : public cv::Algorithm {
  public:
#pragma region Public types

    using Timestamp = std::chrono::system_clock::time_point;
    using Duration = std::chrono::system_clock::duration;
    using Callback = std::function<void(int tag, const Trajectory&)>;

#pragma endregion

#pragma region Public member methods

    /**
     * @brief Construct a new SortTracker object
     *
     * @param maxBBoxAge Age threshold to determine whether a tracked bbox is
     * expired
     * @param minBBoxHitStreak Hit streak threshold to determine whether a
     * tracked bbox can added into its coresponding trajectory
     * @param maxTrajectoryAge Age threshold to determine whether a trajectory
     * is ended
     * @param minTrajectoryNumSamples Sample count threshold to determine
     * whether a trajectory is valid
     * @param minTrajectoryFallingDistance Vertical falling distance threshold
     * to determine whether a trajectory is from a falling object
     * @param iouThreshold IoU threshold for bbox matching across consecutive
     * frames
     * @return
     */
    SortTracker(int maxBBoxAge = 2,
                int minBBoxHitStreak = 3,
                int maxTrajectoryAge = 15,
                int minTrajectoryNumSamples = 16,
                int minTrajectoryFallingDistance = 128.0F,
                float iouThreshold = 0.25F);

    /**
     * @brief Set the callback function of TrajectoryEnded event
     *
     * @param callback Callback function that will be invoked each time a valid
     * trajectory is ended and it will be passed as the argument
     * @return
     */
    void setTrajectoryEndedCallback(Callback callback) {
        _trajectoryEndedCallback = std::move(callback);
    };

    /**
     * @brief Update tracker with new detections
     *
     * @param detections Detected bboxes
     * @param frame Current frame
     * @param timestamp Current timestamp
     * @return
     */
    void update(const std::vector<cv::Rect2f>& detections,
                const cv::Mat& frame,
                Timestamp timestamp = Timestamp::min());

    /**
     * @brief Clear all internal states
     *
     * @return
     */
    void clear() override;

    /**
     * @brief Tells whether this tracker has any trajectory
     *
     * @return  True: No available trajectory
     *          False: Has at least one trajectory
     */
    bool empty() const override;

#pragma endregion

  private:
#pragma region Private types

    /**
     * @brief Predicted {Tag, BBox} pair 
     */
    using Prediction = std::pair<int, cv::Rect2f>;

#pragma endregion

#pragma region Private member variables

    Callback _trajectoryEndedCallback;

    std::map<int, TrackedBBox> _tracks;
    std::map<int, Trajectory> _trajectories;

    std::vector<Prediction> _predictions;
    std::vector<int> _matches;
    std::vector<int> _matchesReversed;

    /**
     * @brief Linear assignment problem solver used for bbox association across frames
     */
    LAPSolver _lapSolver;

    int _maxBBoxAge;
    int _minBBoxHitStreak;
    int _maxTrajectoryAge;
    int _minTrajectoryNumSamples;
    float _minTrajectoryFallingDistance;
    float _iouThreshold;

    int _frameCount;

#pragma endregion

#pragma region Private member methods

    /**
     * @brief Update all tracked bboxes with detections
     * 
     * @param detections Detected bboxes
     * @return  
     */
    void updateTracks(const std::vector<cv::Rect2f>& detections);

    /**
     * @brief Update all trajectories with existing tracked bboxes
     * 
     * @param frame Current frame
     * @param timestamp Current timestamp
     * @return  
     */
    void updateTrajectories(const cv::Mat& frame, Timestamp timestamp);

    /**
     * @brief Tells whether to keep this bbox in the tracker
     *        Conditions that this bbox can be keeped:
     *        1. Recently updated: age < maxAge
     *
     * @param track Tracked bbox
     * @return  True: this bbox should be retained
     *          False: this bbox should be removed
     */
    bool canKeep(const TrackedBBox& track) const;

    /**
     * @brief Tells whether to pick this bbox and add it to its trajectory
     *        Conditions that this bbox can be picked:
     *        1. On hit streak: consecutive hits count >= minHitStrek
     *
     * @param track Tracked bbox
     * @return  True: this bbox can be picked
     *          False: this bbox cannot be picked
     */
    bool canPick(const TrackedBBox& track) const;

    /**
     * @brief Tells whether this trajectory is endded
     *        Conditions that this trajectory is endded:
     *        1. No new sample added for a long time: age >= maxAge
     *
     * @param trajectory Tracked trajectory
     * @return  True: this trajectory is ended
     *          False: this trajectory can still accept future samples
     */
    bool isEnded(const Trajectory& trajectory) const;

    bool isFallingObjectTrajectory(const Trajectory& trajectory) const;

#pragma endregion

#pragma region Static helper methods

    /**
     * @brief Calculate IoU matrix between predicted and detected bboxes
     * 
     * @param predictions 
     * @param detections 
     * @return  
     */
    static cv::Mat getIoU(const std::vector<Prediction>& predictions,
                          const std::vector<cv::Rect2f>& detections);

    /**
     * @brief Allocate an unused tag for new track
     * 
     * @param tracks Current tracked bboxes
     * @return  Unused tag
     */
    static int getUnusedTag(const std::map<int, TrackedBBox>& tracks);

#pragma endregion
};