/**
 * @file vibe_sequential.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Implementation of ViBe++ background subtraction algorithm
 * @version 0.1
 * @date 2021-01-6
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

class ViBeSequential : cv::Algorithm {
  public:
#pragma region Public member methods

    /**
     * @brief Construct a new ViBe algorithm instance
     *
     * @param height Frame height
     * @param width Frame width
     * @param numSamples Number of samples in the background model
     * @param thresholdL1 L1 norm threshold to determine whether a pixel in
     * frame is close to a sample in the background model
     * @param minNumCloseSamples Minimum number of close samples to determine
     * whether a pixel in frame belongs to the background
     * @param updateFactor Set the update rate of background samples (There is
     * 1/updateFactor probability that a sample in the background model will be
     * replaced by current background pixel)
     * @return
     */
    ViBeSequential(int height,
                   int width,
                   int numSamples = 16,
                   uint8_t thresholdL1 = 20,
                   int minNumCloseSamples = 2,
                   int updateFactor = 6);

    /**
     * @brief Construct a new ViBe algorithm instance
     *
     * @param size Frame size
     * @param numSamples Number of samples in the background model
     * @param numHistoryImages Number of history images in the background model
     * @param thresholdL1 L1 norm threshold to determine whether a pixel in
     * frame is close to a sample in the background model
     * @param minNumCloseSamples Minimum number of close samples to determine
     * whether a pixel in frame belongs to the background
     * @param updateFactor Set the update rate of background samples (There is
     * 1/updateFactor probability that a sample in the background model will be
     * replaced by current background pixel)
     * @return
     */
    ViBeSequential(const cv::Size& size,
                   int numSamples = 16,
                   uint8_t thresholdL1 = 20,
                   int minNumCloseSamples = 2,
                   int updateFactor = 6)
        : ViBeSequential(size.height,
                         size.width,
                         numSamples,
                         thresholdL1,
                         minNumCloseSamples,
                         updateFactor) {}

    ~ViBeSequential() override;

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

    /* Parameters */
    int _h;
    int _w;
    int _numPixelsPerFrame;
    int _numSamples;
    uint8_t _thresholdL1;
    int _minNumCloseSamples;
    int _updateFactor;
    bool _isInitalized;

    /* Buffers */
    uint8_t* _historySamples;

    bool _swapHistoryImageFlag;
    uint8_t* _historyImage0;
    uint8_t* _historyImage1;

    std::vector<int> _jump;
    std::vector<int> _neighborIndex;
    std::vector<int> _replaceIndex;

    /* Random generator */
    cv::RNG_MT19937 _rng;

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

    static uint8_t distanceL1(const uint8_t* samplePixel,
                              uint8_t testPixelC0,
                              uint8_t testPixelC1,
                              uint8_t testPixelC2);

    static void
    copyPixel(uint8_t* dst, uint8_t srcC0, uint8_t srcC1, uint8_t srcC2);

    static void swapPixel(uint8_t* pixelA, uint8_t* pixelB);

#pragma endregion
};