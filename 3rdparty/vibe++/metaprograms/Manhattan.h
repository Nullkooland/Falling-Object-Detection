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
#ifndef _LIB_VIBE_XX_COMMON_DISTANCES_MANHATTAN_H_
#define _LIB_VIBE_XX_COMMON_DISTANCES_MANHATTAN_H_

#include "AbsoluteValue.h"
#include "DistanceL1.h"

#define LIB_VIBE_XX_MANHATTAN_MAP_FLOAT_FACTOR(CHANNELS, FACTOR_VAL)           \
    template <>                                                                \
    struct ManhattanFactor<CHANNELS> {                                         \
        static const double FACTOR;                                            \
    };                                                                         \
                                                                               \
    const double ManhattanFactor<CHANNELS>::FACTOR = FACTOR_VAL;

#define LIB_VIBE_XX_MANHATTAN_MAP_UINTG_FACTOR(CHANNELS, FACTOR_VAL)           \
    template <>                                                                \
    struct ManhattanFactor<CHANNELS> {                                         \
        static const uint32_t FACTOR = FACTOR_VAL;                             \
    };

namespace vibe {
namespace internals {
/* ====================================================================== *
 * ManhattanFactor<Channels>                                              *
 * ====================================================================== */

template <uint32_t Channels>
struct ManhattanFactor {};

LIB_VIBE_XX_MANHATTAN_MAP_UINTG_FACTOR(1, 1)
LIB_VIBE_XX_MANHATTAN_MAP_FLOAT_FACTOR(3, 4.5)
} // namespace internals

/* ======================================================================== *
 * Manhattan<Channels>                                                      *
 * ======================================================================== */
template <uint32_t Channels>
struct Manhattan {
    inline static bool
    inRange(const uint8_t* pixel1, const uint8_t* pixel2, uint32_t threshold) {
        return (internals::DistanceL1<Channels>::add(pixel1, pixel2) <=
                (internals::ManhattanFactor<Channels>::FACTOR * threshold));
    }
};
} // namespace vibe

#endif /* _LIB_VIBE_XX_COMMON_DISTANCES_MANHATTAN_H_ */
