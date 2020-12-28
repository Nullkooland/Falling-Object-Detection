#include "lap_solver.hpp"
#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    LAPSolver solver;
    auto costs = std::vector<cv::Mat>();
    // clang-format off

    costs.emplace_back(cv::Matx33f{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9}
    );

    costs.emplace_back(cv::Matx<float, 5, 4>{
        5,  10, 15, 20,
        15, 20, 30, 10,
        10, 20, 15, 30,
        20, 10, 10, 45,
        50, 50, 50, 50,}
    );

    costs.emplace_back(cv::Matx<float, 4, 5>{
        5,  10, 15, 20, 50,
        15, 20, 30, 10, 50,
        10, 20, 15, 30, 50,
        20, 10, 10, 45, 50}
    );

    costs.emplace_back(cv::Matx<float, 20, 8> {
        85, 12, 36, 83, 50, 96, 12,  1,
        84, 35, 16, 17, 40, 94, 16, 52,
        14, 16,  8, 53, 14, 12, 70, 50,
        73, 83, 19, 44, 83, 66, 71, 18,
        36, 45, 29,  4, 61, 15, 70, 47,
         7, 14, 11, 69, 57, 32, 37, 81,
         9, 65, 38, 74, 87, 51, 86, 52,
        52, 40, 56, 10, 42,  2, 26, 36,
        85, 86, 36, 90, 49, 89, 41, 74,
        40, 67,  2, 70, 18,  5, 94, 43,
        85, 12, 36, 83, 50, 96, 12,  1,
        84, 35, 16, 17, 40, 94, 16, 52,
        14, 16, 8 , 53, 14, 12, 70, 50,
        73, 83, 19, 44, 83, 66, 71, 18,
        36, 45, 29,  4, 61, 15, 70, 47,
        7 , 14, 11, 69, 57, 32, 37, 81,
        9 , 65, 38, 74, 87, 51, 86, 52,
        52, 40, 56, 10, 42,  2, 26, 36,
        85, 86, 36, 90, 49, 89, 41, 74,
        40, 67,  2, 70, 18,  5, 94, 43,}
    );


    // clang-format on
    // costs.emplace_back(cv::Matx44f::eye());
    // costs.emplace_back(cv::Matx44f::eye());

    std::vector<int> assignment;
    std::vector<int> assignmentReversed;

    for (const auto& cost : costs) {
        std::printf("\n[COST MATRIX]\n");
        std::cout << cost << std::endl;

        float minTotalCost =
            solver.solve(cost, assignment, assignmentReversed);

        std::printf("\n[MIN TOTAL COST]\n%f\n", minTotalCost);

        std::printf("\n[ASSIGNMENT (TASKS -> WORKERS)]\n");
        for (int i = 0; i < assignment.size(); i++) {
            std::printf("%d -> %d\n", i, assignment[i]);
        }

        std::printf("\n[ASSIGNMENT (WORKERS -> TASKS)]\n");
        for (int j = 0; j < assignmentReversed.size(); j++) {
            std::printf("%d -> %d\n", j, assignmentReversed[j]);
        }
    }

    return 0;
}