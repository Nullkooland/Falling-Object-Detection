/* Copyright - Benjamin Laugraud <blaugraud@ulg.ac.be> - 2016
 * Copyright - Marc Van Droogenbroeck <m.vandroogenbroeck@ulg.ac.be> - 2016
 *
 * ViBe is covered by a patent (see http://www.telecom.ulg.ac.be/research/vibe).
 *
 * Permission to use ViBe without payment of fee is granted for nonprofit
 * educational and research purposes only.
 *
 * This work may not be copied or reproduced in whole or in part for any
 * purpose.
 *
 * Copying, reproduction, or republishing for any purpose shall require a
 * license. Please contact the authors in such cases. All the code is provided
 * without any guarantee.
 */
#include <algorithm>
#include <cstddef>
#include <random>

#include "ViBeBase.h"

using namespace std;
using namespace vibe;

namespace vibe {
inline ostream& operator<<(ostream& os, const ViBeBase& vibe) {
    vibe.print(os);
    return os;
}
} // namespace vibe

/* ========================================================================== *
 * ViBeBase                                                                   *
 * ========================================================================== */

// TODO Replace rand()
// TODO Add noise using generator?
ViBeBase::ViBeBase(int height,
                   int width,
                   uint32_t channels,
                   const uint8_t* buffer)
    : height(height), width(width), numberOfSamples(DEFAULT_NUMBER_OF_SAMPLES),
      matchingThreshold(DEFAULT_MATCHING_THRESHOLD),
      matchingNumber(DEFAULT_MATCHING_NUMBER),
      updateFactor(DEFAULT_UPDATE_FACTOR), stride(width * channels),
      pixels(height * width), numValues(height * width * channels),
      historyImage(nullptr), historyBuffer(nullptr), lastHistoryImageSwapped(),
      jump(nullptr), neighbor(nullptr), position(nullptr) {

    if (height <= 0) {
        throw; // TODO Exception
    }

    if (width <= 0) {
        throw; // TODO Exception
    }

    if (channels <= 0) {
        throw; // TODO Exception
    }

    if (buffer == nullptr) {
        throw; // TODO Exception
    }

    const uint32_t stride = width * channels;

    /* Creates the historyImage structure. */
    historyImage = new uint8_t[NUMBER_OF_HISTORY_IMAGES * stride * height];

    for (uint32_t i = 0; i < NUMBER_OF_HISTORY_IMAGES; ++i) {
        for (int index = stride * height - 1; index >= 0; --index) {
            historyImage[i * stride * height + index] = buffer[index];
        }
    }

    /* Prepare random number generator */
    std::random_device rd;
    this->gen = std::mt19937(rd());
    std::uniform_int_distribution<int> uniformHistoryBuffer(-10, 10);

    /* Now creates and fills the history buffer. */
    historyBuffer = new uint8_t[stride * height *
                                (numberOfSamples - NUMBER_OF_HISTORY_IMAGES)];

    for (int index = stride * height - 1; index >= 0; --index) {
        uint8_t value = buffer[index];

        for (uint32_t x = 0; x < numberOfSamples - NUMBER_OF_HISTORY_IMAGES;
             ++x) {
            historyBuffer[index * (numberOfSamples - NUMBER_OF_HISTORY_IMAGES) +
                          x] =
                min(max(static_cast<int>(value) +
                            uniformHistoryBuffer(this->gen), // Add noise.
                        static_cast<int>(BACKGROUND)),
                    static_cast<int>(FOREGROUND));
        }
    }

    /* Fills the buffers with random values. */
    int size = (width > height) ? 2 * width + 1 : 2 * height + 1;

    jump = new uint32_t[size];
    neighbor = new int[size];
    position = new uint32_t[size];

    std::uniform_int_distribution<int> uniformJump(1, 2 * updateFactor);
    std::uniform_int_distribution<int> uniformNeighbor(-1, 1);
    std::uniform_int_distribution<uint32_t> uniformPosition(0, numberOfSamples -
                                                                   1);

    for (int i = 0; i < size; ++i) {
        /* Values between 1 and 2 * updateFactor. */
        jump[i] = uniformJump(this->gen);
        /* Values between { -width - 1, ... , width + 1 }. */
        neighbor[i] =
            uniformNeighbor(this->gen) * width + uniformNeighbor(this->gen);
        /* Values between 0 and numberOfSamples - 1. */
        position[i] = uniformPosition(this->gen);
    }
}

/******************************************************************************/

ViBeBase::~ViBeBase() {
    delete[] historyImage;
    delete[] historyBuffer;
    delete[] jump;
    delete[] neighbor;
    delete[] position;
}

/******************************************************************************/

uint32_t ViBeBase::getNumberOfSamples() const { return numberOfSamples; }

/******************************************************************************/

uint32_t ViBeBase::getMatchingThreshold() const { return matchingThreshold; }

/******************************************************************************/

void ViBeBase::setMatchingThreshold(int matchingThreshold) {
    if (matchingThreshold <= 0)
        throw; // TODO Exception;

    this->matchingThreshold = matchingThreshold;
}

/******************************************************************************/

uint32_t ViBeBase::getMatchingNumber() const { return matchingNumber; }

/******************************************************************************/

void ViBeBase::setMatchingNumber(int matchingNumber) {
    if (matchingNumber <= 0)
        throw; // TODO Exception;

    this->matchingNumber = matchingNumber;
}

/******************************************************************************/

uint32_t ViBeBase::getUpdateFactor() const { return updateFactor; }

/******************************************************************************/

void ViBeBase::setUpdateFactor(int updateFactor) {
    if (updateFactor <= 0)
        throw; // TODO Exception;

    this->updateFactor = updateFactor;

    /* We also need to change the values of the jump buffer ! */
    int size = 2 * max(width, height) + 1;

    std::uniform_int_distribution<int> uniformJump(1, 2 * updateFactor);
    for (int i = 0; i < size; ++i) {
        // 1 or values between 1 and 2 * updateFactor.
        jump[i] = (updateFactor == 1) ? 1 : uniformJump(gen);
    }
}

/******************************************************************************/

void ViBeBase::print(ostream& os) const {
    os << " - Number of samples per pixel    : " << numberOfSamples << endl;
    os << " - Number of matches needed       : " << matchingNumber << endl;
    os << " - Matching threshold             : " << matchingThreshold << endl;
    os << " - Model update subsampling factor: " << updateFactor;
}
