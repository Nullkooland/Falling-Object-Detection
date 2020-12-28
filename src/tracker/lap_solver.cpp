/**
 * @file lap_solver.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Linear assignment problem solver based on Hungarian algorithm
 * @version 0.1
 * @date 2020-12-15
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "lap_solver.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <opencv2/core.hpp>

float LAPSolver::solve(const cv::Mat& cost,
                       std::vector<int>& assignment,
                       std::vector<int>& assignmentReversed,
                       bool maximize) {
    CV_Assert(cost.type() == CV_32F);

    // Extract number of tasks (rows) and number of workers (columns)
    _m = cost.rows;
    _n = cost.cols;

    // Clear assignment indices
    assignment.resize(_m);
    assignmentReversed.resize(_n);
    std::fill_n(assignment.begin(), _m, -1);
    std::fill_n(assignmentReversed.begin(), _n, -1);

    // If either task or worker set is empty, no assignment could be made
    if (_m == 0 || _n == 0) {
        return 0.0F;
    }

    bool isTransposed = false;
    cv::Mat costTransposed;

    // The algorithm requires more workers than tasks
    // If it's not the case, transpose the cost matrix
    if (_m > _n) {
        costTransposed = cost.t();
        costTransposed.copyTo(_workingCost);

        std::swap(_m, _n);
        isTransposed = true;

    } else {
        cost.copyTo(_workingCost);
    }

    // If the goal is to maximize total cost, minimize on the negative cost
    if (maximize) {
        _workingCost = -_workingCost;
    }

    // Initialize marker table
    if (_markerTable.rows >= _m && _markerTable.cols >= _n) {
        _markerTable({0, _m}, {0, _n}) = Marker::NONE;
    } else {
        _markerTable = cv::Mat(_m, _n, CV_8U, Marker::NONE);
    }

    // Initialize [covered] flags for rows and cols
    _coveredRow.resize(_m);
    _coveredCol.resize(_n);
    std::fill_n(_coveredRow.begin(), _m, false);
    std::fill_n(_coveredCol.begin(), _n, false);

    // Initialize [has-starred-zeros] flags for rows and cols
    _hasStarredZeroInRow.resize(_m);
    _hasStarredZeroInCol.resize(_n);
    std::fill_n(_hasStarredZeroInRow.begin(), _m, false);
    std::fill_n(_hasStarredZeroInCol.begin(), _n, false);

    // Initialize [has-starred-zeros] flags for rows and cols
    _hasNewlyStarredZeroInRow.resize(_m);
    _hasNewlyStarredZeroInCol.resize(_n);
    std::fill_n(_hasNewlyStarredZeroInRow.begin(), _m, false);
    std::fill_n(_hasNewlyStarredZeroInCol.begin(), _n, false);

    // Initialize paths array for augmenting paths algorithm
    _paths.reserve(_m);

    // Do row and column reduction
    // printCost();
    reduceRows();
    // printCost();
    findInitialStarredZeros();

    for (;;) {
        if (coverColsWithStarredZeros() == _m) {
            // Unique assignment for all tasks is available now
            break;
        }

        auto path0 = primeUncoveredZeros();
        // printCost();
        findMaximalMatching(path0);
        // printCost();
    }
    // printCost();

    float minTotalCost = assign(isTransposed ? costTransposed : cost,
                                isTransposed ? assignmentReversed : assignment);

    if (isTransposed) {
        for (int j = 0; j < _m; j++) {
            assignment[assignmentReversed[j]] = j;
        }
    } else {
        for (int i = 0; i < _m; i++) {
            assignmentReversed[assignment[i]] = i;
        }
    }

    return minTotalCost;
}

void LAPSolver::reduceRows() {
    // Row reduction
    for (size_t i = 0; i < _m; i++) {
        auto row = _workingCost.row(i);
        double minVal;

        cv::minMaxLoc(row, &minVal);
        row -= minVal;
    }

    // Column reduction
    // for (size_t j = 0; j < _n; j++) {
    //     auto col = _reducedCost.col(j);
    //     double minVal;

    //     cv::minMaxLoc(col, &minVal);
    //     col -= minVal;
    // }
}

void LAPSolver::findInitialStarredZeros() {
    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            // Try assign uncovered zero
            if (!_hasStarredZeroInCol[j] &&
                _workingCost.at<float>(i, j) == 0.0F) {
                _markerTable.at<Marker>(i, j) = Marker::STAR;

                _hasStarredZeroInRow[i] = true;
                _hasStarredZeroInCol[j] = true;
                break;
            }
        }
    }
}

int LAPSolver::coverColsWithStarredZeros() {
    int numColsCovered = 0;
    for (int j = 0; j < _n; j++) {
        if (_hasStarredZeroInCol[j]) {
            _coveredCol[j] = true;
            numColsCovered++;
        }
    }

    return numColsCovered;
}

std::pair<int, int> LAPSolver::primeUncoveredZeros() {
    auto findUncoveredZero = [this](int& i, int& j) {
        for (int k = i; k < _m; k++) {
            if (_coveredRow[k]) {
                continue;
            }

            for (int l = j; l < _n; l++) {
                if (!_coveredCol[l] && _workingCost.at<float>(k, l) == 0.0F) {
                    // Found an uncovered zero
                    i = k;
                    j = l;
                    return true;
                }
            }
        }

        // All zeros are covered
        return false;
    };

    auto locateStarredZeroInRow = [this](int i) {
        for (int j = 0; j < _n; j++) {
            if (_markerTable.at<Marker>(i, j) == Marker::STAR) {
                return j;
            }
        }

        return -1;
    };

    int i = 0;
    int j = 0;

    for (;;) {
        // printCost();
        // Try to find an uncovered zero
        if (findUncoveredZero(i, j)) {
            // Prime the found zero
            _markerTable.at<Marker>(i, j) = Marker::PRIME;

            if (_hasStarredZeroInRow[i]) {
                j = locateStarredZeroInRow(i);
                _coveredRow[i] = true;
                _coveredCol[j] = false;
            } else {
                return {i, j};
            }
        } else {
            adjustCost();
            i = 0;
            j = 0;
        }
    }
}

void LAPSolver::findMaximalMatching(Path path0) {
    auto locateStarredZeroInCol = [this](int j) {
        for (int i = 0; i < _m; i++) {
            if (_markerTable.at<Marker>(i, j) == Marker::STAR) {
                return i;
            }
        }

        return -1;
    };

    auto locatePrimedZeroInRow = [this](int i) {
        for (int j = 0; j < _n; j++) {
            if (_markerTable.at<Marker>(i, j) == Marker::PRIME) {
                return j;
            }
        }

        return -1;
    };

    _paths.push_back(path0);

    for (;;) {
        auto [i, j] = _paths.back();

        if (_hasStarredZeroInCol[j]) {
            i = locateStarredZeroInCol(j);
            _paths.emplace_back(i, j);
        } else {
            break;
        }

        j = locatePrimedZeroInRow(i);
        _paths.emplace_back(i, j);
    }

    // Augment path
    for (size_t k = 0; k < _paths.size(); k++) {
        auto [i, j] = _paths[k];

        if (k % 2 == 0) {
            // Star the primed zero
            _markerTable.at<Marker>(i, j) = Marker::STAR;

            _hasStarredZeroInRow[i] = true;
            _hasStarredZeroInCol[j] = true;
            _hasNewlyStarredZeroInRow[i] = true;
            _hasNewlyStarredZeroInCol[j] = true;

        } else {
            // Unstar the starred zero
            _markerTable.at<Marker>(i, j) = Marker::NONE;
            if (!_hasNewlyStarredZeroInRow[i]) {
                _hasStarredZeroInRow[i] = false;
            }

            if (!_hasNewlyStarredZeroInCol[j]) {
                _hasStarredZeroInCol[j] = false;
            }
        }
    }

    // Clear paths
    _paths.clear();

    // Clear [has-newly-starred-zero] flags for rows and columns
    std::fill_n(_hasNewlyStarredZeroInRow.begin(), _m, false);
    std::fill_n(_hasNewlyStarredZeroInCol.begin(), _n, false);

    // Erase remaining primes
    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            auto& marker = _markerTable.at<Marker>(i, j);
            if (marker == Marker::PRIME) {
                marker = Marker::NONE;
            }
        }
    }

    // Clear [covered] flags for rows and columns
    std::fill_n(_coveredRow.begin(), _m, false);
    std::fill_n(_coveredCol.begin(), _n, false);
}

void LAPSolver::adjustCost() {
    // Find minimum uncovered cost
    float minUncoveredCost = std::numeric_limits<float>::max();
    for (int i = 0; i < _m; i++) {
        if (_coveredRow[i]) {
            continue;
        }

        for (int j = 0; j < _n; j++) {
            if (!_coveredCol[j]) {
                minUncoveredCost =
                    std::min(_workingCost.at<float>(i, j), minUncoveredCost);
            }
        }
    }

    // Adjust cost matrix
    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            if (_coveredRow[i]) {
                _workingCost.at<float>(i, j) += minUncoveredCost;
            }

            if (!_coveredCol[j]) {
                _workingCost.at<float>(i, j) -= minUncoveredCost;
            }
        }
    }
}

float LAPSolver::assign(const cv::Mat& cost, std::vector<int>& assignment) {
    float minTotalCost = 0.0F;
    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            if (_markerTable.at<Marker>(i, j) == Marker::STAR) {
                assignment[i] = j;
                minTotalCost += cost.at<float>(i, j);
            }
        }
    }

    return minTotalCost;
}

#if defined(DEBUG)
void LAPSolver::printCost() {

    std::printf("\n[COST & MARKER TABLE]\n");

    // Draw column covered line
    for (int j = 0; j < _n; j++) {
        std::printf(_coveredCol[j] ? "  x  " : "     ");
    }
    std::printf("\n");

    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            float cost = _workingCost.at<float>(i, j);
            auto marker = _markerTable.at<Marker>(i, j);
            switch (marker) {
            case Marker::NONE:
                std::printf("%3.0f, ", cost);
                break;
            case Marker::STAR:
                std::printf(" *0, ");
                break;
            case Marker::PRIME:
                std::printf(" \'0, ");
                break;
            }
        }

        if (_coveredRow[i]) {
            std::printf("x");
        }
        std::printf("\n");
        std::fflush(stdout);
    }
}
#endif
