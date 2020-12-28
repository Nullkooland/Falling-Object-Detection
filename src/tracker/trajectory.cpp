/**
 * @file trajectory.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Tracked falling object trajectory
 * @version 0.1
 * @date 2020-12-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#include "trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Trajectory::Trajectory(const cv::Mat& firstFrame) {
    firstFrame.copyTo(_firstFrame);
}

void Trajectory::add(const cv::Rect2f& bbox,
                     const cv::Point2f& velocity,
                     Timestamp timestamp) {
    // clang-format off
    // Pack sample point
    auto sample = SamplePoint {
        .x          = bbox.x, 
        .y          = bbox.y, 
        .width      = bbox.width, 
        .height     = bbox.height,
        .xCenter    = bbox.x + bbox.width / 2, 
        .yCenter    = bbox.y + bbox.height / 2,
        .xVelocity  = velocity.x, 
        .yVelocity  = velocity.y,
        .timestamp  = timestamp,
    };
    // clang-format on

    _samples.push_back(sample);
    // Reset age count
    _age = 0;
}

void Trajectory::draw(cv::Mat& anno) const {
    // Annotate on the first frame this trajectory starts with
    anno = cv::Mat(_firstFrame);

    // Fit parabola
    cv::Vec3f parameters;
    fitParabola(_samples, parameters);

    // Find x coordinate of vertex of the parabola (x = -b/2a)
    // float xVertex = -0.5F * parameters[1] / parameters[0];
    // float xCenterMin = xVertex;
    // float xCenterMax = xVertex;

    float xCenterMin = std::numeric_limits<float>::max();
    float xCenterMax = std::numeric_limits<float>::min();

    // Draw all sample points along this trajectory
    for (const auto& sample : _samples) {
        xCenterMax = std::max(sample.xCenter, xCenterMax);
        xCenterMin = std::min(sample.xCenter, xCenterMin);
        // Draw bbox
        int x = static_cast<int>(sample.x);
        int y = static_cast<int>(sample.y);
        int w = static_cast<int>(sample.width);
        int h = static_cast<int>(sample.height);
        cv::rectangle(anno, {x, y, w, h}, {100, 50, 255});
        // Draw center point
        int xCenter = static_cast<int>(sample.xCenter);
        int yCenter = static_cast<int>(sample.yCenter);
        cv::drawMarker(anno,
                       {xCenter, yCenter},
                       {0, 0, 255},
                       cv::MARKER_TILTED_CROSS,
                       6,
                       2,
                       cv::LINE_AA);
        // Draw velocity vector
        int xVelocityScaled =
            static_cast<int>(VELOCITY_SCALE_FACTOR * sample.xVelocity);
        int yVelocityScaled =
            static_cast<int>(VELOCITY_SCALE_FACTOR * sample.yVelocity);
        cv::arrowedLine(anno,
                        {xCenter, yCenter},
                        {xCenter + xVelocityScaled, yCenter + yVelocityScaled},
                        {0, 255, 0},
                        1,
                        cv::LINE_AA);
    }

    // Generate points along the parabola
    int numSamples =
        static_cast<int>((xCenterMax - xCenterMin) / DRAW_POLYLINE_STEP_X);
    std::vector<cv::Point> points(numSamples);

    for (int i = 0; i < numSamples; i++) {
        float x = xCenterMin + i * DRAW_POLYLINE_STEP_X;
        float y = parameters[0] * x * x + parameters[1] * x + parameters[2];
        points[i].x = static_cast<int>(x);
        points[i].y = static_cast<int>(y);
    }

    // Draw parabola
    cv::polylines(anno, points, false, {0, 255, 255}, 1, cv::LINE_AA);
}

bool Trajectory::fitParabola(const std::vector<SamplePoint>& samples,
                             cv::Vec3f& parameters) {
    int numSamples = samples.size();
    auto A = cv::Mat(numSamples, 3, CV_32F);
    auto b = cv::Mat(numSamples, 1, CV_32F);

    // Fill in matrix A and b
    for (int i = 0; i < numSamples; i++) {
        auto* rowA = A.ptr<float>(i);
        float x = samples[i].xCenter;
        float y = samples[i].yCenter;
        float w = std::exp(static_cast<float>(-i) / numSamples);

        // A[i, *] = [x^2, x, 1]
        rowA[0] = x * x * w;
        rowA[1] = x * w;
        rowA[2] = w;
        // b[i] = y
        b.at<float>(i) = y * w;
    }

    // Solve least-square problem: min|| A · [a, b, c]^T - b ||^2
    return cv::solve(A, b, parameters, cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
}