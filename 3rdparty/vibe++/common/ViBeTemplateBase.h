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
#ifndef _LIB_VIBE_XX_COMMON_VIBE_TEMPLATE_BASE_H_
#define _LIB_VIBE_XX_COMMON_VIBE_TEMPLATE_BASE_H_

#include <iostream>

#include "ViBeBase.h"

namespace vibe {
template <class Derived>
class ViBeTemplateBase : public ViBeBase {
  protected:
    using Base = ViBeBase;

    ViBeTemplateBase(int height,
                     int width,
                     uint32_t channels,
                     const uint8_t* buffer);

  public:
    virtual ~ViBeTemplateBase() {}

    inline void segmentation(const uint8_t* buffer, uint8_t* segmentationMap);

    inline void update(const uint8_t* buffer, uint8_t* updatingMask);
};

#include "ViBeTemplateBase.t"
} // namespace vibe

#endif /* _LIB_VIBE_XX_COMMON_VIBE_TEMPLATE_BASE_H_ */
