/**
 * @file vibe.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief OpenCV implementation of ViBe background subtraction algorithm with
 * SIMD and parallel optimization
 * @version 0.1
 * @date 2020-12-30
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <array>
#include <limits>
#include <opencv2/core.hpp>
#include <utility>

class ViBe : public cv::Algorithm {
  public:
    ViBe(int width,
         int height,
         uint8_t thresholdL1 = 20,
         uint32_t minNumCloseSamples = 2,
         int updateFactor = 6);

    void segment(const cv::Mat& frame, cv::Mat& fgMask);
    void update(const cv::Mat& frame, cv::Mat& updateMask);

    void clear() override;

    bool empty() const override { return !_isInitalized; }

  private:
#pragma region Private constants

    static constexpr int NUM_SAMPLES = 16;

    static constexpr std::array<std::pair<int, int>, 8> OFFSET_8_NEIGHBOR{
        // clang-format off
        std::make_pair(-1, -1),
        std::make_pair(-1,  0),
        std::make_pair(-1, +1),
        std::make_pair( 0, -1),
        std::make_pair( 0, +1),
        std::make_pair(+1, -1),
        std::make_pair(+1,  0),
        std::make_pair(+1, +1),
        // clang-format on
    };

    static constexpr uint8_t BACKGROUND_LABEL =
        std::numeric_limits<uint8_t>::min();

    static constexpr uint8_t FOREGROUND_LABEL =
        std::numeric_limits<uint8_t>::max();

#pragma endregion

    cv::Mat _historyImages;
    cv::Mat _samples;
    cv::Mat _randomTable;

    bool _isInitalized;
    int _h;
    int _w;
    uint8_t _thresholdL1;
    uint32_t _minNumCloseSamples;
    int _updateFactor;

    void init(const cv::Mat& frame);

    static uint32_t countCloseSamples(const uint8_t* testPixel,
                                      const uint8_t* samples,
                                      uint8_t threshold);
};