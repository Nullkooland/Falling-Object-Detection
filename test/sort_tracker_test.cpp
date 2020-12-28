#include "tracked_bbox.hpp"
#include "tracker.hpp"
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    auto fs =
        cv::FileStorage("data/gen_bboxes.json",
                        cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    auto input = cv::Mat();
    auto node = fs["gen_bboxes"];
    node >> input;

    auto output = cv::Mat(3, input.size, CV_32F);

    std::cout << input.size << std::endl;

    auto tracker = SortTracker(3, 3);
    auto tracks = std::vector<TrackedBBox>();
    auto detections = std::vector<cv::Rect2f>();
    detections.reserve(8);

    auto frame = cv::Mat(720, 1280, CV_8UC3);
    auto videoWriter = cv::VideoWriter(
        "data/tracker_test.mp4", cv::VideoWriter::fourcc('h', '2', '6', '4'),
        30, {1280, 720});

    for (int t = 0; t < input.size[0]; t++) {
        for (int k = 0; k < input.size[1]; k++) {
            auto* bbox = input.ptr<float>(t, k);
            int x = static_cast<int>(bbox[0]);
            int y = static_cast<int>(bbox[1]);
            int w = static_cast<int>(bbox[2]);
            int h = static_cast<int>(bbox[3]);
            cv::rectangle(frame, {x, y, w, h}, {0, 255, 0}, 2);
            detections.emplace_back(bbox[0], bbox[1], bbox[2], bbox[3]);

            cv::putText(frame, std::to_string(k), {x, y - 8},
                        cv::FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0}, 1,
                        cv::LINE_AA);
        }

        tracker.update(detections, tracks);
        detections.clear();

        for (const auto& bbox : tracks) {
            auto r = bbox.getRect();
            int x = static_cast<int>(r.x);
            int y = static_cast<int>(r.y);
            int w = static_cast<int>(r.width);
            int h = static_cast<int>(r.height);
            cv::rectangle(frame, {x, y, w, h}, {50, 0, 255}, 2);

            int tag = bbox.getTag();
            cv::putText(frame, std::to_string(tag), {x, y - 8},
                        cv::FONT_HERSHEY_PLAIN, 1.0, {50, 0, 255}, 1,
                        cv::LINE_AA);
        }

        cv::imshow("SORT Test", frame);
        videoWriter << frame;
        if (cv::waitKey(30) == static_cast<int>('q')) {
            break;
        }

        frame = 0;
        cv::putText(frame, "Measurement", {64, 36}, cv::FONT_HERSHEY_PLAIN, 2.0,
                    {0, 255, 0}, 2, cv::LINE_AA);
        cv::putText(frame, "Tracking", {64, 72}, cv::FONT_HERSHEY_PLAIN, 2.0,
                    {50, 0, 255}, 2, cv::LINE_AA);
    }

    videoWriter.release();
    return 0;
}