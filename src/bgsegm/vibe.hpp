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

/**
 * @brief ViBe background substractor with SIMD and parallel optimization
 */
class ViBe : public cv::Algorithm {
  public:
#pragma region Public member methods

    /**
     * @brief Construct a new ViBe algorithm instance
     *
     * @param height Frame height
     * @param width Frame width
     * @param thresholdL1 L1 norm threshold to determine whether a pixel in
     * frame is close to a sample in the background model
     * @param minNumCloseSamples Minimum number of close samples to determine
     * whether a pixel in frame belongs to the background
     * @param updateFactor Set the update rate of background samples (There is
     * 1/updateFactor probability that a sample in the background model will be
     * replaced by current background pixel)
     * @return
     */
    ViBe(int height,
         int width,
         uint8_t thresholdL1 = 20,
         uint32_t minNumCloseSamples = 2,
         int updateFactor = 6);

    /**
     * @brief Construct a new ViBe algorithm instance
     *
     * @param size Frame size
     * @param thresholdL1 L1 norm threshold to determine whether a pixel in
     * frame is close to a sample in the background model
     * @param minNumCloseSamples Minimum number of close samples to determine
     * whether a pixel in frame belongs to the background
     * @param updateFactor Set the update rate of background samples (There is
     * 1/updateFactor probability that a sample in the background model will be
     * replaced by current background pixel)
     * @return
     */
    ViBe(const cv::Size& size,
         uint8_t thresholdL1 = 20,
         uint32_t minNumCloseSamples = 2,
         int updateFactor = 6)
        : ViBe(size.height,
               size.width,
               thresholdL1,
               minNumCloseSamples,
               updateFactor) {}

    /**
     * @brief
     *
     * @param frame Input current frame (in CV_8UC3 format)
     * @param fgMask Output foreground mask (in CV_8UC1 format)
     * @return
     */
    void segment(const cv::Mat& frame, cv::Mat& fgMask);

    /**
     * @brief
     *
     * @param frame Input current frame (in CV_8UC3 format)
     * @param updateMask Input update mask (in CV_8UC1 format)
     * @return
     */
    void update(const cv::Mat& frame, const cv::Mat& updateMask);

    /**
     * @brief Reset the background substractor by invalidating all samples in
     * the background model
     *
     * @return
     */
    void clear() override;

    /**
     * @brief Tells whether the ViBe bg substractor is initialized (with inital
     * samples in the background model)
     *
     * @return  True: ViBe is initialized
     *          False: ViBe is uninitialized
     */
    bool empty() const override { return !_isInitalized; }

#pragma endregion
  private:
#pragma region Private constants

    /**
     * @brief Number of samples in the background model
     */
    static constexpr int NUM_SAMPLES = 16;

    /**
     * @brief XY position offsets for 8-neighbors of a pixel
     */
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

    /**
     * @brief Label value indicating a background pixel in the output mask
     */
    static constexpr uint8_t BACKGROUND_LABEL =
        std::numeric_limits<uint8_t>::min();

    /**
     * @brief Label value indicating a foreground pixel in the output mask
     */
    static constexpr uint8_t FOREGROUND_LABEL =
        std::numeric_limits<uint8_t>::max();

#pragma endregion
#pragma region Private member variables

    /**
     * @brief Background model samples, in [H x W x N x C] layout
     */
    cv::Mat _samples;

    /**
     * @brief Random numbers used in model update, in [H x W x 3] layout,
     * each of the three random numbers in an element represents:
     * [0]: updateIndicator         \in [0, updateFactor)
     * [1]: indexOfSampleToReplace  \in [0, NUM_SAMPLES)
     * [2]: indexOfNeighborToUpdate \in [0, 8)
     */
    cv::Mat _randomTable;

    bool _isInitalized;
    int _h;
    int _w;
    uint8_t _thresholdL1;
    uint32_t _minNumCloseSamples;
    int _updateFactor;

#pragma endregion

#pragma region Private member methods

    /**
     * @brief Initialize the ViBe bg substractor
     *
     * @param frame Initial frame
     * @return
     */
    void init(const cv::Mat& frame);

#pragma endregion

#pragma region Static helper methods

    /**
     * @brief Find number of samples in the background model that are close to
     * test pixel
     *
     * @param testPixel Test pixel from current frame (CV_8UC3 x 1)
     * @param samples Sample pixels from the background model (CV_8UC3 x 16)
     * @param threshold L1 threshold to determine whether a sample pixel is
     * close to the test pixel
     * @return
     */
    static uint32_t countCloseSamples(const uint8_t* testPixel,
                                      const uint8_t* samples,
                                      uint8_t threshold);

#pragma endregion
};