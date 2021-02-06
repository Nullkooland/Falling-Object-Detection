/**
 * @file tracker.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Falling object bbox tracker based on SORT algorithm
 * @version 0.1
 * @date 2020-12-27
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "tracker.hpp"

#include "kalman_filter.hpp"
#include "tracked_bbox.hpp"
#include "trajectory.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
namespace chrono = std::chrono;

SortTracker::SortTracker(int maxBBoxAge,
                         int minBBoxHitStreak,
                         int maxTrajectoryAge,
                         int minTrajectoryNumSamples,
                         int minTrajectoryFallingDistance,
                         float iouThreshold)
    : _maxBBoxAge(maxBBoxAge),
      _minBBoxHitStreak(minBBoxHitStreak),
      _maxTrajectoryAge(maxTrajectoryAge),
      _minTrajectoryNumSamples(minTrajectoryNumSamples),
      _minTrajectoryFallingDistance(minTrajectoryFallingDistance),
      _iouThreshold(iouThreshold),
      _tagCount(0),
      _frameCount(0) {}

void SortTracker::update(const std::vector<cv::Rect2f>& detections,
                         const cv::Mat& frame,
                         Timestamp timestamp) {

    updateTracks(detections);
    updateTrajectories(frame, timestamp);

    _frameCount++;
}

void SortTracker::updateTracks(const std::vector<cv::Rect2f>& detections) {
    // No tracked bbox available, make all detected bboxes as tracked
    if (_tracks.empty()) {
        for (const auto& bbox : detections) {
            int tag = getUnusedTag();
            _tracks.emplace(tag, TrackedBBox(bbox));
        }
        return;
    }

    // Predict tracked bboxes
    _predictions.reserve(_tracks.size());
    _predictions.clear();

    for (auto& [tag, bbox] : _tracks) {
        _predictions.emplace_back(tag, bbox.predict({0.05F, 0.7F}));
    }

    // Initialize matches index table (prediction -> detections)
    _matches.resize(_predictions.size(), -1);

    // Calculate IoU matrix
    auto iou = getIoU(_predictions, detections);
    // Solve for optimal matches that can maximize total sum of IoUs
    _lapSolver.solve(iou, _matches, _matchesReversed, true);

    // std::cout << "\n[IoU Matrix]\n" << iou << std::endl;
    // std::cout << "\n[Assignment]" << std::endl;

    // for (int i = 0; i < _matches.size(); i++) {
    //     std::printf("%d -> %d\n", i, _matches[i]);
    // }

    // Update matched tracks and remove expired tracks
    for (int i = 0; i < _matches.size(); i++) {
        int j = _matches[i];
        int tag = _predictions[i].first;
        auto it = _tracks.find(tag);
        auto& [_, track] = *it;

        if (j != -1) {
            // Good match, update the track
            if (iou.at<float>(i, j) > _iouThreshold) {
                track.update(detections[j]);
                continue;
            }
            // Poor match is canceled
            _matchesReversed[j] = -1;
        }

        // Remove expired/bad track
        if (!canKeep(track)) {
            _tracks.erase(it);
            // If the track is removed,
            // its coresponding trajectory will end immediately
            if (auto jt = _trajectories.find(tag); jt != _trajectories.end()) {
                auto& [_, trajectory] = *jt;
                trajectory.incrementAge(_maxTrajectoryAge + 1);
            }
        }
    }

    // Add unmatched detections to tracks
    for (int j = 0; j < _matchesReversed.size(); j++) {
        if (_matchesReversed[j] == -1) {
            int tag = getUnusedTag();
            _tracks.emplace(tag, TrackedBBox(detections[j]));
        }
    }
}

void SortTracker::updateTrajectories(const cv::Mat& frame,
                                     Timestamp timestamp) {
    if (timestamp == Timestamp::min()) {
        timestamp = chrono::system_clock::now();
    }

    for (const auto& [tag, track] : _tracks) {
        if (!canPick(track)) {
            continue;
        }

        auto it = _trajectories.find(tag);
        // No trajectory for this track, create a new one
        if (it == _trajectories.end()) {
            auto [itNew, isSuccessful] =
                _trajectories.emplace(tag, Trajectory(frame));
            it = itNew;
        }

        // Add the track to its coresponding trajectory
        auto& [_, trajectory] = *it;
        trajectory.add(track.getRect(), track.getVelocity(), timestamp);
    }

    for (auto it = _trajectories.begin(); it != _trajectories.end();) {
        auto [tag, trajectory] = *it;
        // Save and remove ended trajectory
        if (isEnded(trajectory)) {
            if (isFallingObjectTrajectory(trajectory)) {
                _trajectoryEndedCallback(tag, trajectory);
            }
            it = _trajectories.erase(it);
        } else {
            it = std::next(it);
        }

        trajectory.incrementAge();
    }
}

void SortTracker::clear() {
    _tracks.clear();
    _trajectories.clear();
}

bool SortTracker::empty() const { return _trajectories.empty(); }

bool SortTracker::canKeep(const TrackedBBox& track) const {
    return track.getAge() <= _maxBBoxAge;
}

bool SortTracker::canPick(const TrackedBBox& track) const {
    return track.getHitStreak() >= _minBBoxHitStreak;
}

bool SortTracker::isEnded(const Trajectory& trajectory) const {
    return trajectory.getAge() > _maxTrajectoryAge;
}

bool SortTracker::isFallingObjectTrajectory(
    const Trajectory& trajectory) const {

    if (int numSamples = trajectory.getNumSamples();
        numSamples < _minTrajectoryNumSamples) {
        std::printf("[INVALID TRAJECTORY] Number of samples: %d\n", numSamples);
        return false;
    }

    if (float fallDistance = trajectory.getRangeY();
        fallDistance < _minTrajectoryFallingDistance) {
        std::printf("[INVALID TRAJECTORY] Falling distance: %.2f\n",
                    fallDistance);
        return false;
    }

    return true;
}

int SortTracker::getUnusedTag() { return _tagCount++; }

cv::Mat SortTracker::getIoU(const std::vector<Prediction>& predictions,
                            const std::vector<cv::Rect2f>& detections) {
    int m = predictions.size();
    int n = detections.size();
    auto cost = cv::Mat(m, n, CV_32F);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto bboxDetected = detections[j];
            auto bboxPredicted = predictions[i].second;
            auto bboxIntersected = bboxPredicted & bboxDetected;

            float areaA = bboxDetected.area();
            float areaB = bboxPredicted.area();
            float areaI = bboxIntersected.area();

            float iou = 0.0F;
            if (!bboxIntersected.empty()) {
                iou = areaI / (areaA + areaB - areaI);
            }

            cost.at<float>(i, j) = iou;
        }
    }

    return cost;
}
