/**
 * @file utils.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Utilities
 * @version 0.1
 * @date 2020-12-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>
#include <type_traits>

class Utils {
  public:
    /**
     * @brief Get random BGR colors
     * 
     * @tparam Num Number of colors
     * @tparam Type Data type
     * @return  [Num, 3] color matrix
     */
    template <size_t Num = 32, typename Type = uint8_t>
    static constexpr cv::Matx<Type, Num, 3> getRandomColors() {
        cv::Matx<Type, Num, 3> colors;

        if constexpr (std::is_integral_v<Type>) {
            for (size_t i = 0; i < Num; i++) {
                colors(i, 0) = static_cast<Type>(
                    cv::theRNG().uniform(B_RANGE_LOW, B_RANGE_HIGH) *
                    std::numeric_limits<Type>::max());
                
                colors(i, 1) = static_cast<Type>(
                    cv::theRNG().uniform(G_RANGE_LOW, G_RANGE_HIGH) *
                    std::numeric_limits<Type>::max());
                
                colors(i, 2) = static_cast<Type>(
                    cv::theRNG().uniform(R_RANGE_LOW, R_RANGE_HIGH) *
                    std::numeric_limits<Type>::max());
            }
        } else if (std::is_floating_point_v<Type>) {
            for (size_t i = 0; i < Num; i++) {
                colors(i, 0) = cv::theRNG().uniform(B_RANGE_LOW, B_RANGE_HIGH);
                colors(i, 1) = cv::theRNG().uniform(G_RANGE_LOW, G_RANGE_HIGH);
                colors(i, 2) = cv::theRNG().uniform(R_RANGE_LOW, R_RANGE_HIGH);
            }
        }

        return colors;
    }

  private:
    static constexpr float R_RANGE_HIGH = 0.8F;
    static constexpr float R_RANGE_LOW = 0.1F;
    static constexpr float G_RANGE_HIGH = 0.9F;
    static constexpr float G_RANGE_LOW = 0.05F;
    static constexpr float B_RANGE_HIGH = 0.8F;
    static constexpr float B_RANGE_LOW = 0.1F;
};