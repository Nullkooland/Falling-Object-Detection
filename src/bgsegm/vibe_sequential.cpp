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

#include <array>
#include <opencv2/core.hpp>

ViBeSequential::ViBeSequential(int _h,
                               int _w,
                               int numSamples,
                               uint8_t thresholdL1,
                               int minNumCloseSamples,
                               int updateFactor)
    : _h(_h),
      _w(_w),
      _numPixelsPerFrame(_h * _w),
      _numSamples(numSamples),
      _thresholdL1(thresholdL1),
      _minNumCloseSamples(minNumCloseSamples),
      _updateFactor(updateFactor) {

    // Allocate buffers
    _historyImage0 = static_cast<uint8_t*>(cv::fastMalloc(_h * _w * 3));
    _historyImage1 = static_cast<uint8_t*>(cv::fastMalloc(_h * _w * 3));
    _historySamples =
        static_cast<uint8_t*>(cv::fastMalloc(_h * _w * numSamples * 3));

    int size = (_w > _h) ? 2 * _w + 1 : 2 * _h + 1;
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

    /* Compare with first history image */
    for (int i = 0; i < numPixels; i++) {
        uint8_t pixelC0 = frame.data[i * 3 + 0];
        uint8_t pixelC1 = frame.data[i * 3 + 1];
        uint8_t pixelC2 = frame.data[i * 3 + 2];
        if (distanceL1(_historyImage0 + i * 3, pixelC0, pixelC1, pixelC2) >=
            _thresholdL1) {
            fgMask.data[i] = _minNumCloseSamples;
        }
    }

    /* Compare with second history image */
    for (int i = 0; i < numPixels; i++) {
        uint8_t pixelC0 = frame.data[i * 3 + 0];
        uint8_t pixelC1 = frame.data[i * 3 + 1];
        uint8_t pixelC2 = frame.data[i * 3 + 2];
        if (distanceL1(_historyImage1 + i * 3, pixelC0, pixelC1, pixelC2) <
            _thresholdL1) {
            fgMask.data[i]--;
        }
    }

    for (int i = 0; i < numPixels; i++) {
        if (fgMask.data[i] > 0) {
            /* We need to check the full border and swap values with the first
             * or second historyImage. We still need to find a match before we
             * can stop our search.
             */
            uint8_t* historySample = _historySamples + i * _numSamples * 3;
            uint8_t pixelC0 = frame.data[i * 3 + 0];
            uint8_t pixelC1 = frame.data[i * 3 + 1];
            uint8_t pixelC2 = frame.data[i * 3 + 2];

            for (int k = 0; k < _numSamples; k++) {
                if (distanceL1(
                        historySample + k * 3, pixelC0, pixelC1, pixelC2) <
                    _thresholdL1) {
                    fgMask.data[i]--;

                    /* Swaping: Putting found value in history image buffer. */
                    swapPixel(swappingHistoryImage + i * 3,
                              historySample + k * 3);

                    /* Exit inner loop. */
                    if (fgMask.data[i] <= 0) {
                        break;
                    }
                }
            }
        }
    }

    /* Produces the output. Note that this step is application-dependent. */
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

    /* All the frame, except the border. */
    int shift;
    int indX;
    int indY;
    int y;
    int x;
    int k;

    for (y = 1; y < _h - 1; ++y) {
        shift = _rng.uniform(0, _w);
        indX = _jump[shift]; // index_jump should never be zero (> 1).
        k = _replaceIndex[shift];
        int neighborIndex = _neighborIndex[shift];

        while (indX < _w - 1) {
            int i = indX + y * _w;
            uint8_t pixelC0 = frame.data[i * 3 + 0];
            uint8_t pixelC1 = frame.data[i * 3 + 1];
            uint8_t pixelC2 = frame.data[i * 3 + 2];

            if (updateMask.data[i] == BACKGROUND_LABEL) {
                /* In-place substitution. */

                if (k < 2) {
                    uint8_t* historyImage =
                        (k == 0) ? _historyImage0 : _historyImage1;

                    copyPixel(historyImage + i * 3, pixelC0, pixelC1, pixelC2);
                    copyPixel(historyImage + (i + neighborIndex) * 3,
                              pixelC0,
                              pixelC1,
                              pixelC2);
                } else {
                    int kSample = k - 2;

                    copyPixel(_historySamples +
                                  (i * _numSamples * 3 + kSample * 3),
                              pixelC0,
                              pixelC1,
                              pixelC2);

                    copyPixel(_historySamples +
                                  ((i + neighborIndex) * _numSamples * 3 +
                                   kSample * 3),
                              pixelC0,
                              pixelC1,
                              pixelC2);
                }
            }

            ++shift;
            indX += _jump[shift];
        }
    }

    auto replaceSample =
        [this](const cv::Mat& frame, const cv::Mat& updateMask, int i, int k) {
            uint8_t pixelC0 = frame.data[i * 3 + 0];
            uint8_t pixelC1 = frame.data[i * 3 + 1];
            uint8_t pixelC2 = frame.data[i * 3 + 2];

            if (updateMask.data[i] == BACKGROUND_LABEL) {
                if (k < 2) {
                    uint8_t* historyImage =
                        (k == 0) ? _historyImage0 : _historyImage1;
                    copyPixel(historyImage + i * 3, pixelC0, pixelC1, pixelC2);
                } else {
                    int kSample = k - 2;
                    copyPixel(_historySamples +
                                  (i * _numSamples * 3 + kSample * 3),
                              pixelC0,
                              pixelC1,
                              pixelC2);
                }
            }
        };

    /* First row. */
    y = 0;
    shift = _rng.uniform(0, _w);
    indX = _jump[shift]; // index_jump should never be zero (> 1).
    k = _replaceIndex[shift];

    while (indX <= _w - 1) {
        int i = indX + y * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indX += _jump[shift];
    }

    /* Last row. */
    y = _h - 1;
    shift = _rng.uniform(0, _w);
    indX = _jump[shift]; // index_jump should never be zero (> 1).
    k = _replaceIndex[shift];

    while (indX <= _w - 1) {
        int i = indX + y * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indX += _jump[shift];
    }

    /* First column. */
    x = 0;
    shift = _rng.uniform(0, _h);
    indY = _jump[shift]; // index_jump should never be zero (> 1).
    k = _replaceIndex[shift];

    while (indY <= _h - 1) {
        int i = x + indY * _w;

        replaceSample(frame, updateMask, i, k);

        ++shift;
        indY += _jump[shift];
    }

    /* Last column. */
    x = _w - 1;
    shift = _rng.uniform(0, _h);
    indY = _jump[shift]; // index_jump should never be zero (> 1).
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

    // Fill random i tables
    for (int i = 0; i < _replaceIndex.size(); i++) {
        _jump[i] = _rng.uniform(1, _updateFactor * 2 + 1);
        _replaceIndex[i] = _rng.uniform(0, _numSamples);
        _neighborIndex[i] = _rng.uniform(-1, 2);
    }

    _isInitalized = true;
}

uint8_t ViBeSequential::distanceL1(const uint8_t* samplePixel,
                                   uint8_t testPixelC0,
                                   uint8_t testPixelC1,
                                   uint8_t testPixelC2) {
    // clang-format off
    return std::abs(testPixelC0 - samplePixel[0]) +
           std::abs(testPixelC1 - samplePixel[1]) +
           std::abs(testPixelC2 - samplePixel[2]);
    // clang-format on
}

void ViBeSequential::copyPixel(uint8_t* dst,
                               uint8_t srcC0,
                               uint8_t srcC1,
                               uint8_t srcC2) {
    dst[0] = srcC0;
    dst[1] = srcC1;
    dst[2] = srcC2;
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