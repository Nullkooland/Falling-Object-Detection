/**
 * @file vibe.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief OpenCV implementation of ViBe background subtraction algorithm with
 * SIMD and parallel optimization
 * @version 0.1
 * @date 2020-12-30
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "vibe.hpp"

#include <array>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/operations.hpp>

ViBe::ViBe(int height,
           int width,
           uint8_t thresholdL1,
           uint32_t minNumCloseSamples,
           int updateFactor)
    : _isInitalized(false),
      _h(height),
      _w(width),
      _thresholdL1(thresholdL1),
      _minNumCloseSamples(minNumCloseSamples),
      _updateFactor(updateFactor) {

    _samples = cv::Mat({_h, _w, NUM_SAMPLES}, CV_8UC3);
    _randomTable = cv::Mat(_h, _w, CV_8UC3);
}

void ViBe::segment(const cv::Mat& frame, cv::Mat& fgMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!fgMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(fgMask.rows == _h && fgMask.cols == _w);

    if (!_isInitalized) {
        init(frame);
    }

    fgMask.setTo(BACKGROUND_LABEL);

    cv::parallel_for_(
        {0, _h * _w}, [this, &frame, &fgMask](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++) {
                int i = r / _w;
                int j = r % _w;
                const auto* pixel = frame.ptr<uint8_t>(i, j);
                auto* samples = _samples.ptr<uint8_t>(i, j);
                if (countCloseSamples(pixel, samples, _thresholdL1) <
                    _minNumCloseSamples) {
                    fgMask.at<uint8_t>(i, j) = FOREGROUND_LABEL;
                }
            }
        });
}

void ViBe::update(const cv::Mat& frame, const cv::Mat& updateMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!updateMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(updateMask.rows == _h && updateMask.cols == _w);

    cv::randu(_randomTable,
              std::array<int, 3>{0, 0, 0},
              std::array<int, 3>{_updateFactor, NUM_SAMPLES, 8});

    cv::parallel_for_(
        {0, _h * _w}, [this, &frame, &updateMask](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++) {
                int i = r / _w;
                int j = r % _w;
                const auto* randomIndex = _randomTable.ptr<int8_t>(i, j);

                if (updateMask.at<uint8_t>(i, j) != BACKGROUND_LABEL ||
                    randomIndex[0] != 0) {
                    continue;
                }

                const auto& pixel = frame.at<cv::Vec3b>(i, j);
                auto* samples = _samples.ptr<cv::Vec3b>(i, j);

                uint8_t k = randomIndex[1];
                samples[k] = pixel;

                auto [iOffset, jOffset] = OFFSET_8_NEIGHBOR[randomIndex[2]];
                i += iOffset;
                j += jOffset;
                i = i < 0 ? 0 : (i >= _h ? _h - 1 : i);
                j = j < 0 ? 0 : (j >= _w ? _w - 1 : j);

                auto* neighborSamples = _samples.ptr<cv::Vec3b>(i, j);
                neighborSamples[k] = pixel;
            }
        });
}

void ViBe::init(const cv::Mat& frame) {
    cv::setRNGSeed(std::time(nullptr));
    // Fill in samples matrix
    cv::parallel_for_({0, _h * _w}, [this, &frame](const cv::Range& range) {
        for (int r = range.start; r < range.end; r++) {
            int i = r / _w;
            int j = r % _w;
            const auto& pixel = frame.at<cv::Vec3b>(i, j);
            auto* samples = _samples.ptr<cv::Vec3b>(i, j);

            for (int k = 0; k < NUM_SAMPLES; k++) {
                samples[k][0] = cv::saturate_cast<uint8_t>(
                    pixel[0] + cv::theRNG().uniform(-12, 12));
                samples[k][1] = cv::saturate_cast<uint8_t>(
                    pixel[1] + cv::theRNG().uniform(-12, 12));
                samples[k][2] = cv::saturate_cast<uint8_t>(
                    pixel[2] + cv::theRNG().uniform(-12, 12));
            }
        }
    });

    _isInitalized = true;
}

void ViBe::clear() { _isInitalized = false; }

uint32_t ViBe::countCloseSamples(const uint8_t* testPixel,
                                 const uint8_t* samples,
                                 uint8_t threshold) {
    // Unpack 16 sample pixels to to vectors in 3 channels
    cv::v_uint8x16 vSamplesC0;
    cv::v_uint8x16 vSamplesC1;
    cv::v_uint8x16 vSamplesC2;
    cv::v_load_deinterleave(samples, vSamplesC0, vSamplesC1, vSamplesC2);

    // Duplicate test pixel to vectors in 3 channels
    cv::v_uint8x16 vPixelC0 = cv::v_setall_u8(testPixel[0]);
    cv::v_uint8x16 vPixelC1 = cv::v_setall_u8(testPixel[1]);
    cv::v_uint8x16 vPixelC2 = cv::v_setall_u8(testPixel[2]);

    // Calculate absdiff of 3 channels respectively
    cv::v_uint8x16 vAbsDiffC0 = cv::v_absdiff(vPixelC0, vSamplesC0);
    cv::v_uint8x16 vAbsDiffC1 = cv::v_absdiff(vPixelC1, vSamplesC1);
    cv::v_uint8x16 vAbsDiffC2 = cv::v_absdiff(vPixelC2, vSamplesC2);

    // Sum across channels to get L1 norm vector
    cv::v_uint8x16 vL1Norm = vAbsDiffC0 + vAbsDiffC1 + vAbsDiffC2;
    // Duplicate threshold to vector
    cv::v_uint8x16 vThreshold = cv::v_setall_u8(threshold);
    // Count number of sample pixels that close to test pixel
    uint32_t numSamplesInRange = cv::v_reduce_sum(vL1Norm < vThreshold);

    return numSamplesInRange;
}