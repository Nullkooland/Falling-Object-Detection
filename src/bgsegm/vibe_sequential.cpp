/**
 * @file vibe_sequential.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Implementation of ViBe++ background subtraction algorithm
 * @version 0.2
 * @date 2021-01-14
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "vibe_sequential.hpp"

#include <array>
#include <opencv2/core.hpp>

ViBeSequential::ViBeSequential(int height,
                               int width,
                               int numSamples,
                               uint32_t thresholdL1,
                               int minNumCloseSamples,
                               int updateFactor)
    : _h(height),
      _w(width),
      _numPixelsPerFrame(height * width),
      _numSamples(numSamples),
      _thresholdL1(thresholdL1 * 3),
      _minNumCloseSamples(minNumCloseSamples),
      _updateFactor(updateFactor),
      _swapHistoryImageFlag(false),
      _isInitalized(false) {

    // Allocate buffers
    _historyImage0 = static_cast<uint8_t*>(cv::fastMalloc(height * width * 3));
    _historyImage1 = static_cast<uint8_t*>(cv::fastMalloc(height * width * 3));
    _historySamples =
        static_cast<uint8_t*>(cv::fastMalloc(height * width * numSamples * 3));

    int size = (width > height) ? 2 * width + 1 : 2 * height + 1;
    _jump.resize(size);
    _neighborIndex.resize(size);
    _replaceIndex.resize(size);
}

ViBeSequential::~ViBeSequential() {
    cv::fastFree(_historyImage0);
    cv::fastFree(_historyImage1);
    cv::fastFree(_historySamples);
}

void ViBeSequential::segment(const cv::Mat& frame, cv::Mat& fgMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!fgMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(fgMask.rows == _h && fgMask.cols == _w);
    CV_Assert(frame.isContinuous());
    CV_Assert(fgMask.isContinuous());

    if (!_isInitalized) {
        init(frame);
    }

    // Clear segmentation mask
    fgMask.setTo(_minNumCloseSamples - 1);

    int numPixels = _h * _w;

    _swapHistoryImageFlag = !_swapHistoryImageFlag;
    uint8_t* swappingHistoryImage =
        _swapHistoryImageFlag ? _historyImage1 : _historyImage0;

    // Compare with first history image
    for (int i = 0; i < numPixels; i++) {
        if (!isClose(
                _historyImage0 + i * 3, frame.data + i * 3, _thresholdL1)) {
            fgMask.data[i] = _minNumCloseSamples;
        }
    }

    // Compare with second history image
    for (int i = 0; i < numPixels; i++) {
        if (isClose(_historyImage1 + i * 3, frame.data + i * 3, _thresholdL1)) {
            fgMask.data[i]--;
        }
    }

    // Compare with history samples
    for (int i = 0; i < numPixels; i++) {
        // This pixel is already labelled as background, move to next one
        if (fgMask.data[i] < 0) {
            continue;
        }

        uint8_t* historySample = _historySamples + i * _numSamples * 3;
        std::array<uint8_t, 3> currentPixel;
        copyPixel(currentPixel.data(), frame.data + i * 3);

        for (int k = 0; k < _numSamples && fgMask.data[i] > 0; k++) {
            if (isClose(
                    historySample + k * 3, currentPixel.data(), _thresholdL1)) {
                fgMask.data[i]--;

                // Put the close sample pixel into history image buffer
                swapPixel(swappingHistoryImage + i * 3, historySample + k * 3);
            }
        }
    }

    // Assgin foreground label for "survivors"
    for (int i = 0; i < numPixels; i++) {
        if (fgMask.data[i] > 0) {
            fgMask.data[i] = FOREGROUND_LABEL;
        }
    }
}

void ViBeSequential::update(const cv::Mat& frame, const cv::Mat& updateMask) {
    CV_Assert(!frame.empty());
    CV_Assert(!updateMask.empty());
    CV_Assert(frame.rows == _h && frame.cols == _w);
    CV_Assert(updateMask.rows == _h && updateMask.cols == _w);
    CV_Assert(frame.isContinuous());
    CV_Assert(updateMask.isContinuous());

    int shift;
    int indX;
    int indY;
    int y;
    int x;
    int k;

    // Update background model
    // All but border
    for (y = 1; y < _h - 1; ++y) {
        shift = _rng.uniform(0, _w);
        indX = _jump[shift];
        k = _replaceIndex[shift];
        int neighborIndex = _neighborIndex[shift];

        while (indX < _w - 1) {
            int i = indX + y * _w;
            std::array<uint8_t, 3> currentPixel;
            copyPixel(currentPixel.data(), frame.data + i * 3);

            if (updateMask.data[i] == BACKGROUND_LABEL) {
                if (k < 2) {
                    uint8_t* historyImage =
                        (k == 0) ? _historyImage0 : _historyImage1;

                    copyPixel(historyImage + i * 3, currentPixel.data());
                    copyPixel(historyImage + (i + neighborIndex) * 3,
                              currentPixel.data());
                } else {
                    int kSample = k - 2;

                    copyPixel(_historySamples +
                                  (i * _numSamples * 3 + kSample * 3),
                              currentPixel.data());

                    copyPixel(_historySamples +
                                  ((i + neighborIndex) * _numSamples * 3 +
                                   kSample * 3),
                              currentPixel.data());
                }
            }

            ++shift;
            indX += _jump[shift];
        }
    }

    auto replaceSample =
        [this](const cv::Mat& frame, const cv::Mat& updateMask, int i, int k) {
            if (updateMask.data[i] == BACKGROUND_LABEL) {
                if (k < 2) {
                    uint8_t* historyImage =
                        (k == 0) ? _historyImage0 : _historyImage1;
                    copyPixel(historyImage + i * 3, frame.data + i * 3);
                } else {
                    int kSample = k - 2;
                    copyPixel(_historySamples +
                                  (i * _numSamples * 3 + kSample * 3),
                              frame.data + i * 3);
                }
            }
        };

    // First row
    y = 0;
    shift = _rng.uniform(0, _w);
    indX = _jump[shift];
    k = _replaceIndex[shift];

    while (indX <= _w - 1) {
        int i = indX + y * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indX += _jump[shift];
    }

    // Last row
    y = _h - 1;
    shift = _rng.uniform(0, _w);
    indX = _jump[shift];
    k = _replaceIndex[shift];

    while (indX <= _w - 1) {
        int i = indX + y * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indX += _jump[shift];
    }

    // First column
    x = 0;
    shift = _rng.uniform(0, _h);
    indY = _jump[shift];
    k = _replaceIndex[shift];

    while (indY <= _h - 1) {
        int i = x + indY * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indY += _jump[shift];
    }

    // Last column
    x = _w - 1;
    shift = _rng.uniform(0, _h);
    indY = _jump[shift];
    k = _replaceIndex[shift];

    while (indY <= _h - 1) {
        int i = x + indY * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indY += _jump[shift];
    }
}

void ViBeSequential::clear() { _isInitalized = false; }

void ViBeSequential::init(const cv::Mat& frame) {
    // Fill in history images
    uint8_t* src = frame.data;
    int sizePerFrame = _numPixelsPerFrame * 3;
    std::copy(src, src + sizePerFrame, _historyImage0);
    std::copy(src, src + sizePerFrame, _historyImage1);

    // Fill in inital background samples
    int sizePerSamplesBatch = _numSamples * 3;
    for (int i = 0; i < _numPixelsPerFrame; i++, src += 3) {
        uint8_t* dst = _historySamples + i * sizePerSamplesBatch;
        for (int k = 0; k < _numSamples; k++, dst += 3) {
            dst[0] = cv::saturate_cast<uint8_t>(src[0] + _rng.uniform(-10, 10));
            dst[1] = cv::saturate_cast<uint8_t>(src[1] + _rng.uniform(-10, 10));
            dst[2] = cv::saturate_cast<uint8_t>(src[2] + _rng.uniform(-10, 10));
        }
    }

    // Fill random indices tables
    for (int i = 0; i < _replaceIndex.size(); i++) {
        _jump[i] = _rng.uniform(1, _updateFactor * 2 + 1);
        _replaceIndex[i] = _rng.uniform(0, _numSamples);
        _neighborIndex[i] = _rng.uniform(-1, 2);
    }

    _isInitalized = true;
}

bool ViBeSequential::isClose(const uint8_t* pixelA,
                             const uint8_t* pixelB,
                             uint32_t thresholdL1) {

    uint32_t normL1 = static_cast<uint32_t>(std::abs(pixelA[0] - pixelB[0])) +
                      static_cast<uint32_t>(std::abs(pixelA[1] - pixelB[1])) +
                      static_cast<uint32_t>(std::abs(pixelA[2] - pixelB[2]));
    return normL1 <= thresholdL1;
}

void ViBeSequential::copyPixel(uint8_t* dst, const uint8_t* src) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
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