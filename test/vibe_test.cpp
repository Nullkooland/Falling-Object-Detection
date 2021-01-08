#include "vibe_sequential.hpp"
// #include "vibe.hpp"

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

constexpr auto VIDEO_PATH = "data/apartment.264";
// constexpr auto VIDEO_PATH = "data/office_building_floor_15.avi";

int main(int argc, char* argv[]) {
    cv::setNumThreads(8);

    auto cap = cv::VideoCapture(1);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameCount = 0;

    auto vibe =
        std::make_unique<ViBeSequential>(height, width, 14, 20, 2, 6);
    // auto vibe = std::make_unique<ViBe>(height, width, 25, 3, 8);

    auto frame = cv::Mat(height, width, CV_8UC3);
    auto fgMask = cv::Mat(height, width, CV_8UC1);
    auto updateMask = cv::Mat(height, width, CV_8UC1);
    auto se3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});

    // Prepare runtime measurement
    int64_t tickBegin;
    int64_t tickEnd;
    auto getTimespanMs = [](int64_t tickBegin, int64_t tickEnd) {
        return static_cast<double>(tickEnd - tickBegin) /
               cv::getTickFrequency() * 1e3;
    };

    std::array<char, 64> str;
    cv::namedWindow("frame");
    cv::namedWindow("fgmask");

    while (cap.read(frame)) {
        frameCount++;
        if (frameCount < 16) {
            continue;
        }

        tickBegin = cv::getTickCount();

        vibe->segment(frame, fgMask);

        tickEnd = cv::getTickCount();
        double timeSegmentMs = getTimespanMs(tickBegin, tickEnd);

        cv::morphologyEx(fgMask, updateMask, cv::MORPH_OPEN, se3x3);

        tickBegin = cv::getTickCount();

        vibe->update(frame, updateMask);

        tickEnd = cv::getTickCount();
        double timeUpdateMs = getTimespanMs(tickBegin, tickEnd);

        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, se3x3);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, se3x3);

        std::sprintf(str.data(),
                     "ViBe process time: [SEGMENT] %.2f ms, [UPDATE] %.2f ms",
                     timeSegmentMs,
                     timeUpdateMs);

        cv::putText(frame,
                    str.data(),
                    {12, 36},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    {50, 0, 255},
                    2,
                    cv::LINE_AA);

        cv::imshow("frame", frame);
        cv::imshow("fgmask", fgMask);

        if (cv::waitKey(16) == static_cast<int>('q')) {
            break;
        }

        if (frameCount % 30 == 0) {
            std::printf("[FRAME #%-4d] %s\n", frameCount, str.data());
        }
    }

    cv::destroyAllWindows();
    return 0;
}