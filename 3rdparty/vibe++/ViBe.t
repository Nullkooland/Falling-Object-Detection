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
#ifdef _LIB_VIBE_XX_VIBE_H_

#include <random>

/* ========================================================================== *
 * ViBeSequential<Channels, Distance>                                         *
 * ========================================================================== */

template <uint32_t Channels, class Distance>
ViBeSequential<Channels, Distance>::ViBeSequential(int height,
                                                   int width,
                                                   const uint8_t* buffer)
    : Base(height, width, Channels, buffer) {}

/******************************************************************************/

template <uint32_t Channels, class Distance>
void ViBeSequential<Channels, Distance>::_CRTP_segmentation(
    const uint8_t* buffer, uint8_t* segmentationMap) {
#ifndef NDEBUG
    if (buffer == NULL)
        throw; // TODO Exception

    if (segmentationMap == NULL)
        throw; // TODO Exception
#endif         /* NDEBUG */

    /* Even though those variables/contents are redundant with the variables of
     * ViBeBase, they avoid additional dereference instructions.
     */

    uint32_t pixels = this->pixels;
    uint32_t numValues = this->numValues;
    uint32_t matchingNumber = this->matchingNumber;
    uint32_t matchingThreshold = this->matchingThreshold;

    uint8_t* historyImage = this->historyImage;
    uint8_t* historyBuffer = this->historyBuffer;

    /* Initialize segmentation map. */
    memset(segmentationMap, matchingNumber - 1, pixels);

    /* First history Image structure. */
    for (int index = pixels - 1; index >= 0; --index) {
        if (!Distance::inRange(buffer + (Channels * index),
                               historyImage + (Channels * index),
                               matchingThreshold))
            segmentationMap[index] = matchingNumber;
    }

    /* Next historyImages. */
    for (uint32_t i = 1; i < this->NUMBER_OF_HISTORY_IMAGES; ++i) {
        uint8_t* pels = historyImage + i * numValues;

        for (int index = pixels - 1; index >= 0; --index) {
            if (Distance::inRange(buffer + (Channels * index),
                                  pels + (Channels * index), matchingThreshold))
                --segmentationMap[index];
        }
    }

    /* For swapping. */
    this->lastHistoryImageSwapped =
        (this->lastHistoryImageSwapped + 1) % this->NUMBER_OF_HISTORY_IMAGES;

    uint8_t* swappingImageBuffer =
        historyImage + (this->lastHistoryImageSwapped) * numValues;

    /* Now, we move in the buffer and leave the historyImages. */
    int numberOfTests =
        (this->numberOfSamples - this->NUMBER_OF_HISTORY_IMAGES);

    for (int index = pixels - 1; index >= 0; --index) {
        if (segmentationMap[index] > 0) {
            /* We need to check the full border and swap values with the first
             * or second historyImage. We still need to find a match before we
             * can stop our search.
             */
            uint32_t indexHistoryBuffer = (Channels * index) * numberOfTests;
            uint8_t currentValue[Channels];

            internals::CopyPixel<Channels>::copy(&(currentValue[0]),
                                                 buffer + (Channels * index));

            for (int i = numberOfTests; i > 0;
                 --i, indexHistoryBuffer += Channels) {
                if (Distance::inRange(&(currentValue[0]),
                                      historyBuffer + indexHistoryBuffer,
                                      matchingThreshold)) {
                    --segmentationMap[index];

                    /* Swaping: Putting found value in history image buffer. */
                    internals::SwapPixels<Channels>::swap(
                        swappingImageBuffer + (Channels * index),
                        historyBuffer + indexHistoryBuffer);

                    /* Exit inner loop. */
                    if (segmentationMap[index] <= 0)
                        break;
                }
            }
        }
    }

    /* Produces the output. Note that this step is application-dependent. */
    for (uint8_t* mask = segmentationMap; mask < segmentationMap + pixels;
         ++mask) {
        if (*mask > 0)
            *mask = this->FOREGROUND;
    }
}

/******************************************************************************/

template <uint32_t Channels, class Distance>
void ViBeSequential<Channels, Distance>::_CRTP_update(const uint8_t* buffer,
                                                      uint8_t* updatingMask) {
#ifndef NDEBUG
    if (buffer == NULL)
        throw; // TODO Exception

    if (updatingMask == NULL)
        throw; // TODO Exception
#endif         /* NDEBUG */

    /* Some variables. */
    uint32_t height = this->height;
    uint32_t width = this->width;
    uint32_t numValues = this->numValues;

    uint8_t* historyImage = this->historyImage;
    uint8_t* historyBuffer = this->historyBuffer;

    /* Some utility variable. */
    int numberOfTests =
        (this->numberOfSamples - this->NUMBER_OF_HISTORY_IMAGES);

    uint32_t* jump = this->jump;
    int* neighbor = this->neighbor;
    uint32_t* position = this->position;

    /* All the frame, except the border. */
    uint32_t shift, indX, indY;
    uint32_t x, y;

    std::uniform_int_distribution<uint32_t> uniformCols(0, width - 1);
    std::uniform_int_distribution<uint32_t> uniformRows(0, height - 1);

    for (y = 1; y < height - 1; ++y) {
        shift = uniformCols(this->gen);
        indX = jump[shift]; // index_jump should never be zero (> 1).

        while (indX < width - 1) {
            int index = indX + y * width;

            if (updatingMask[index] == this->BACKGROUND) {
                /* In-place substitution. */
                uint8_t currentValue[Channels];

                internals::CopyPixel<Channels>::copy(
                    &(currentValue[0]), buffer + (Channels * index));

                int indexNeighbor = Channels * (index + neighbor[shift]);

                if (position[shift] < this->NUMBER_OF_HISTORY_IMAGES) {
                    internals::CopyPixel<Channels>::copy(
                        historyImage +
                            (Channels * index + position[shift] * numValues),
                        &(currentValue[0]));

                    internals::CopyPixel<Channels>::copy(
                        historyImage +
                            (indexNeighbor + position[shift] * numValues),
                        &(currentValue[0]));
                } else {
                    int pos = position[shift] - this->NUMBER_OF_HISTORY_IMAGES;

                    internals::CopyPixel<Channels>::copy(
                        historyBuffer + ((Channels * index) * numberOfTests +
                                         Channels * pos),
                        &(currentValue[0]));

                    internals::CopyPixel<Channels>::copy(
                        historyBuffer +
                            (indexNeighbor * numberOfTests + Channels * pos),
                        &(currentValue[0]));
                }
            }

            ++shift;
            indX += jump[shift];
        }
    }

    /* First row. */
    y = 0;
    shift = uniformCols(this->gen);
    indX = jump[shift]; // index_jump should never be zero (> 1).

    while (indX <= width - 1) {
        int index = indX + y * width;

        if (updatingMask[index] == this->BACKGROUND) {
            if (position[shift] < this->NUMBER_OF_HISTORY_IMAGES) {
                internals::CopyPixel<Channels>::copy(
                    historyImage +
                        (Channels * index + position[shift] * numValues),
                    buffer + (Channels * index));
            } else {
                int pos = position[shift] - this->NUMBER_OF_HISTORY_IMAGES;

                internals::CopyPixel<Channels>::copy(
                    historyBuffer +
                        ((Channels * index) * numberOfTests + Channels * pos),
                    buffer + (Channels * index));
            }
        }

        ++shift;
        indX += jump[shift];
    }

    /* Last row. */
    y = height - 1;
    shift = uniformCols(this->gen);
    indX = jump[shift]; // index_jump should never be zero (> 1).

    while (indX <= width - 1) {
        int index = indX + y * width;

        if (updatingMask[index] == this->BACKGROUND) {
            if (position[shift] < this->NUMBER_OF_HISTORY_IMAGES) {
                internals::CopyPixel<Channels>::copy(
                    historyImage +
                        (Channels * index + position[shift] * numValues),
                    buffer + (Channels * index));
            } else {
                int pos = position[shift] - this->NUMBER_OF_HISTORY_IMAGES;

                internals::CopyPixel<Channels>::copy(
                    historyBuffer +
                        ((Channels * index) * numberOfTests + Channels * pos),
                    buffer + (Channels * index));
            }
        }

        ++shift;
        indX += jump[shift];
    }

    /* First column. */
    x = 0;
    shift = uniformRows(this->gen);
    indY = jump[shift]; // index_jump should never be zero (> 1).

    while (indY <= height - 1) {
        int index = x + indY * width;

        if (updatingMask[index] == this->BACKGROUND) {
            if (position[shift] < this->NUMBER_OF_HISTORY_IMAGES) {
                internals::CopyPixel<Channels>::copy(
                    historyImage +
                        (Channels * index + position[shift] * numValues),
                    buffer + (Channels * index));
            } else {
                int pos = position[shift] - this->NUMBER_OF_HISTORY_IMAGES;

                internals::CopyPixel<Channels>::copy(
                    historyBuffer +
                        ((Channels * index) * numberOfTests + Channels * pos),
                    buffer + (Channels * index));
            }
        }

        ++shift;
        indY += jump[shift];
    }

    /* Last column. */
    x = width - 1;
    shift = uniformRows(this->gen);
    indY = jump[shift]; // index_jump should never be zero (> 1).

    while (indY <= height - 1) {
        int index = x + indY * width;

        if (updatingMask[index] == this->BACKGROUND) {
            if (position[shift] < this->NUMBER_OF_HISTORY_IMAGES) {
                internals::CopyPixel<Channels>::copy(
                    historyImage +
                        (Channels * index + position[shift] * numValues),
                    buffer + (Channels * index));
            } else {
                int pos = position[shift] - this->NUMBER_OF_HISTORY_IMAGES;

                internals::CopyPixel<Channels>::copy(
                    historyBuffer +
                        ((Channels * index) * numberOfTests + Channels * pos),
                    buffer + (Channels * index));
            }
        }

        ++shift;
        indY += jump[shift];
    }

    std::uniform_int_distribution<uint32_t> uniformUpdate(
        0, this->updateFactor - 1);
    std::uniform_int_distribution<uint32_t> uniformNumSamples(
        0, this->numberOfSamples - 1);

    /* The first pixel! */
    if (uniformUpdate(this->gen) == 0) {
        if (updatingMask[0] == 0) {
            uint32_t position = uniformNumSamples(this->gen);

            if (position < this->NUMBER_OF_HISTORY_IMAGES) {
                internals::CopyPixel<Channels>::copy(
                    historyImage + (position * numValues), buffer);
            } else {
                int pos = position - this->NUMBER_OF_HISTORY_IMAGES;

                internals::CopyPixel<Channels>::copy(
                    historyBuffer + (Channels * pos), buffer);
            }
        }
    }
}

#endif /* _LIB_VIBE_XX_VIBE_H_ */
