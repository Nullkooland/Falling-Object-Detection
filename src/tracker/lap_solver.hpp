/**
 * @file lap_solver.hpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Linear assignment problem solver based on Hungarian algorithm
 * @version 0.1
 * @date 2020-12-18
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Linear Assignment Problem (LAP) Solver
 *        based on Kuhn-Munkres algorithm (Hungarian algorithm)
 *        See https://brc2.com/the-algorithm-workshop for details
 *
 */
class LAPSolver final {
  public:
#pragma region Public member methods

    /**
     * @brief Solve a linear assignment problem instance,
     *        the default goal is to minimize total cost
     *
     * @param cost Cost matrix (tasks x workers)
     * @param assignment Assignment indices (tasks -> workers)
     * @param assignmentReversed Reversed assignment indices (workers -> tasks)
     * @param maximize Set the goal to maximize total cost
     * @return  Optimal total cost
     */
    float solve(const cv::Mat& cost,
                std::vector<int>& assignment,
                std::vector<int>& assignmentReversed,
                bool maximize = false);

#pragma endregion

  private:
#pragma region Private types

    /**
     * @brief Marker for cost value (can be stored as uint8_t)
     */
    enum Marker : uint8_t {
        NONE,
        STAR,
        PRIME,
    };

    /**
     * @brief Augment path
     */
    using Path = std::pair<int, int>;

#pragma endregion

#pragma region Private member variables

    /**
     * @brief Copy of the input cost matrix. It will be modified during the
     * algorithm
     */
    cv::Mat _workingCost;

    /**
     * @brief 2D table that holds markers corresponding to each cost value
     */
    cv::Mat _markerTable;

    /**
     * @brief Size of the cost matrix (mxn: number of tasks x number of workers)
     */
    int _m, _n;

    /**
     * @brief 1D mask to tell whether some rows in the cost matrix is covered by
     * a line
     */
    std::vector<bool> _coveredRow;

    /**
     * @brief 1D mask to tell whether some columns in the cost matrix is covered
     * by a line
     */
    std::vector<bool> _coveredCol;

    /**
     * @brief 1D mask to tell whether some rows in the cost matrix contain zero
     * with STAR marker
     */
    std::vector<bool> _hasStarredZeroInRow;

    /**
     * @brief 1D mask to tell whether some columns in the cost matrix contain
     * zero with STAR marker
     */
    std::vector<bool> _hasStarredZeroInCol;

    /**
     * @brief 1D mask to tell whether some rows in the cost matrix contain zero
     * with marker that just turned from PRIME to STAR in the
     * 'findMaximalMatching' step (Step 5)
     */
    std::vector<bool> _hasNewlyStarredZeroInRow;

    /**
     * @brief 1D mask to tell whether some columns in the cost matrix contain
     * zero with marker that just turned from PRIME to STAR in the
     * 'findMaximalMatching' step (Step 5)
     */
    std::vector<bool> _hasNewlyStarredZeroInCol;

    /**
     * @brief Paths used in the 'findMaximalMatching' step (Step 5)
     */
    std::vector<Path> _paths;

#pragma endregion

#pragma region Private member methods
    /**
     * @brief Step 1: row cost reduction
     *
     * @return
     */
    void reduceRows();

    /**
     * @brief Step 2: star uncovered zeros in each row
     *
     * @return
     */
    void findInitialStarredZeros();

    /**
     * @brief Step 3: cover columns with zeros having STAR marker, if all
     * columns are covered, an unique assignment is available
     *
     * @return Number of columns covered
     */
    int coverColsWithStarredZeros();

    /**
     * @brief Step 4: find uncovered zeros and prime it, until
     * there is no starred zero in the row containing the lastet primed zero
     *
     * @return position of the primed zero where this method ended
     */
    Path primeUncoveredZeros();

    /**
     * @brief Step 5: solve maximal matching problem with augmenting path
     * algorithm
     *
     * @param path0 Initial path
     * @return
     */
    void findMaximalMatching(Path path0);

    /**
     * @brief Step 6: cost adjustment for step 4
     *
     * @return
     */
    void adjustCost();

    /**
     * @brief Do optimal assignment
     *
     * @param cost Cost matrix
     * @param assignment Assignment indices (tasks -> workers)
     * @return  Optimal total cost
     */
    float assign(const cv::Mat& cost, std::vector<int>& assignment);

#if defined(DEBUG)
    /**
     * @brief Print working cost matrix (DEBUG only)
     *
     * @return
     */
    void printCost();
#endif

#pragma endregion
};