/**
 * @file kalman_filter_instantiation.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Instantiation of templated Kalman filter
 * @version 0.1
 * @date 2020-12-15
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once
#include "kalman_filter.hpp"

// Need this disgusting workaround to avoid link error
template class KalmanFilter<7, 4, 2>; // For bbox tracker
template class KalmanFilter<4, 2, 1>; // For test