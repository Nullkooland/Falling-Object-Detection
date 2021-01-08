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

#include "vibe_sequential.hpp"

#include <_types/_uint8_t.h>
#include <cstddef>
#include <cstdio>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/saturate.hpp>
#include <opencv2/highgui.hpp>

ViBeSequential::ViBeSequential(int height,
                               int width,
                               int numSamples,
                               int numHistoryImages,
                               uint8_t thresholdL1,
                               int minNumCloseSamples,
                               int updateFactor)
    : _h(height),
      _w(width),
      _numPixelsPerFrame(height * width),
      _numSamples(numSamples),
      _numHistoryImages(numHistoryImages),
      _thresholdL1(thresholdL1),
      _minNumCloseSamples(minNumCloseSamples),
      _updateFactor(updateFactor) {

    // Allocate buffers
    _historyImages = static_cast<uint8_t*>(
        cv::fastMalloc(numHistoryImages * height * width * 3));

    _samples =
        static_cast<uint8_t*>(cv::fastMalloc(height * width * numSamples * 3));

    int size = (width > height) ? 2 * width + 1 : 2 * height + 1;
    _jump.resize(size);
    _neighborIndex.resize(size);
    _replaceIndex.resize(size);
}

ViBeSequential::~ViBeSequential() {
    cv::fastFree(_historyImages);
    cv::fastFree(_samples);
}

void ViBeSequential::segment(const cv::Mat& frame, cv::Mat& fgMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!fgMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(fgMask.rows == _h && fgMask.cols == _w);
    // CV_Assert(frame.isContinuous());
    CV_Assert(fgMask.isContinuous());

    if (!_isInitalized) {
        init(frame);
    }

    // Clear segmentation mask
    fgMask.setTo(BACKGROUND_LABEL);

    int sizePerFrame = _numPixelsPerFrame * 3;
    int sizePerSamplesBatch = _numSamples * 3;

    _lastSwappedImageIndex = (_lastSwappedImageIndex + 1) % _numHistoryImages;
    uint8_t* swappingHistoryImage =
        _historyImages + _lastSwappedImageIndex * sizePerFrame;

    // Assgin foreground labels using background model
    for (int i = 0; i < _numPixelsPerFrame; i++) {
        // Read current pixel
        uint8_t testPixelC0 = frame.data[i * 3];
        uint8_t testPixelC1 = frame.data[i * 3 + 1];
        uint8_t testPixelC2 = frame.data[i * 3 + 2];
        // Compare with first history image
        if (uint8_t d = distanceL1(
                testPixelC0, testPixelC1, testPixelC2, _historyImages + i * 3);
            d >= _thresholdL1) {
            fgMask.data[i] = FOREGROUND_LABEL;
            continue;
        }

        int numCloseSamples = 0;

        // Compare with remianing history images
        for (int k = 1;
             k < _numHistoryImages && numCloseSamples < _minNumCloseSamples;
             k++) {
            uint8_t* currentHistoryImagePixel =
                _historyImages + k * sizePerFrame + i;
            if (distanceL1(testPixelC0,
                           testPixelC1,
                           testPixelC2,
                           currentHistoryImagePixel) < _thresholdL1) {
                numCloseSamples++;
            }
        }

        // Compare with history samples
        for (int k = 0;
             k < _numSamples && numCloseSamples < _minNumCloseSamples;
             k++) {
            uint8_t* currentSamplePixel =
                _samples + i * sizePerSamplesBatch + k * 3;

            if (distanceL1(
                    testPixelC0, testPixelC1, testPixelC2, currentSamplePixel) <
                _thresholdL1) {
                numCloseSamples++;

                // uint8_t* currentSwappingHistoryImagePixel =
                //     swappingHistoryImage + i * 3;
                // swapPixel(currentSamplePixel, currentSwappingHistoryImagePixel);
            }
        }

        if (numCloseSamples < _minNumCloseSamples) {
            fgMask.data[i] = FOREGROUND_LABEL;
        }
    }
}

void ViBeSequential::update(const cv::Mat& frame, const cv::Mat& updateMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!updateMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(updateMask.rows == _h && updateMask.cols == _w);
    // CV_Assert(frame.isContinuous());
    CV_Assert(updateMask.isContinuous());
}

void ViBeSequential::clear() { _isInitalized = false; }

void ViBeSequential::init(const cv::Mat& frame) {
    // Fill in history images
    uint8_t* src = frame.data;
    int sizePerFrame = _numPixelsPerFrame * 3;
    for (int k = 0; k < _numHistoryImages; k++) {
        uint8_t* dst = _historyImages + k * sizePerFrame;
        std::copy(src, src + sizePerFrame, dst);
    }

    // Fill in inital background samples
    int sizePerSamplesBatch = _numSamples * 3;
    for (int i = 0; i < _numPixelsPerFrame; i++, src += 3) {
        uint8_t* dst = _samples + i * sizePerSamplesBatch;
        for (int k = 0; k < _numSamples; k++, dst += 3) {
            dst[0] = cv::saturate_cast<uint8_t>(src[0] + _rng.uniform(-10, 10));
            dst[1] = cv::saturate_cast<uint8_t>(src[1] + _rng.uniform(-10, 10));
            dst[2] = cv::saturate_cast<uint8_t>(src[2] + _rng.uniform(-10, 10));
        }
    }

    // Fill random index tables
    for (int i = 0; i < _replaceIndex.size(); i++) {
        _jump[i] = _rng.uniform(1, _updateFactor * 2 + 1);
        _replaceIndex[i] = _rng.uniform(0, _numSamples);
        _neighborIndex[i] = _rng.uniform(-1, 2);
    }

    _isInitalized = true;
}

uint8_t ViBeSequential::distanceL1(uint8_t testPixelC0,
                                   uint8_t testPixelC1,
                                   uint8_t testPixelC2,
                                   const uint8_t* samplePixel) {
    // clang-format off
    return std::abs(testPixelC0 - samplePixel[0]) +
           std::abs(testPixelC1 - samplePixel[1]) +
           std::abs(testPixelC2 - samplePixel[2]);
    // clang-format on
}

void ViBeSequential::swapPixel(uint8_t* pixelA, uint8_t* pixelB) {
    uint8_t tempC0 = pixelA[0];
    uint8_t tempC1 = pixelA[1];
    uint8_t tempC2 = pixelA[2];

    pixelA[0] = pixelB[0];
    pixelA[1] = pixelB[1];
    pixelA[2] = pixelB[2];

    pixelB[0] = tempC0;
    pixelB[1] = tempC1;
    pixelB[2] = tempC2;
}