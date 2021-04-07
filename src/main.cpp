/**
 * @file main.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Program entry
 * @version 0.3
 * @date 2021-01-14
 *
 * @copyright Copyright (c) 2020
 *
 */
#include "tracker.hpp"
#include "trajectory.hpp"
#include "utils.hpp"
#include "vibe_sequential.hpp"
#include "video_reader.hpp"

#include <argparse/argparse.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#if !defined(ROCKCHIP_PLATFORM)
#include <opencv2/highgui.hpp>
#endif

/**
 * @brief Setup command line arguments
 *
 * @return argparse::ArgumentParser arg parser
 */
static argparse::ArgumentParser getArgParser() {
    auto parser = argparse::ArgumentParser("RTSP client demo with ffmpeg");

    // clang-format off
    parser.add_argument("-v", "--verbose")
        .help("Print verbose debug message")
        .default_value(false)
        .implicit_value(true);

    parser.add_argument("-l", "--local")
        .help("Print verbose debug message")
        .default_value(false)
        .implicit_value(true);

    parser.add_argument("-a", "--addr")
        .required()
        .help("RTSP address")
        .default_value(std::string("localhost"));

    parser.add_argument("-f", "--file")
        .required()
        .help("RTSP file or stream")
        .default_value(std::string("cam"));

    parser.add_argument("-p", "--port")
        .help("RTSP port")
        .default_value(554)
        .action([](const std::string& arg) {return std::stoi(arg);} );

    parser.add_argument("-u", "--user")
        .help("RTSP username")
        .default_value(std::string(""));

    parser.add_argument("-k", "--password")
        .help("RTSP password")
        .default_value(std::string(""));

    parser.add_argument("--rtsp_transport")
        .help("RTSP transport protocol")
        .default_value(std::string("tcp"));

    parser.add_argument("--buffer_size")
        .help("FFMPEG buffer size (bytes)")
        .default_value(1024UL * 64)
        .action([](const std::string& arg) { return std::stoi(arg); });

    parser.add_argument("--rotate")
        .help("Rotate clockwise angle in degrees (can only be set to 0, 90, 180, 270)")
        .default_value(0)
        .action([](const std::string& arg) {
            int rotateAngle = std::stoi(arg);
            switch (rotateAngle) {
            case 90: return cv::RotateFlags::ROTATE_90_CLOCKWISE;
            case 180: return cv::RotateFlags::ROTATE_180;
            case 270: return cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE;
            default: return static_cast<cv::RotateFlags>(-1);
            }
        });

    parser.add_argument("--resize")
        .help("Resize to given height")
        .default_value(cv::Size(0, 0))
        .action([](const std::string& arg) { 
            size_t delimterPos = arg.find_first_of('x');
            if (delimterPos == std::string::npos) {
                return cv::Size(0, 0);
            }

            int width = std::stoi(arg.substr(0, delimterPos));
            int height = std::stoi(arg.substr(delimterPos + 1));
            return cv::Size(width, height);
        });

    parser.add_argument("--log")
        .help("Log tracked objects to file")
        .default_value("falling_objects_detection_log.json");

    parser.add_argument("--log_interval")
        .help("Number of frames between two logs")
        .default_value(0UL)
        .action([](const std::string& arg) { return std::stoul(arg); });

    parser.add_argument("-o", "--output")
        .help("Output directory")
        .default_value(std::string("data"));

    parser.add_argument("--max_blob_count")
        .help("Max number of detected foreground blobs in a valid frame")
        .default_value(64)
        .action([](const std::string& arg) { return std::stoi(arg); });

    // clang-format on

    return parser;
}

int main(int argc, char* argv[]) {
    auto parser = getArgParser();

    // Parse command line arguments
    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& e) {
        std::printf("%s\n", e.what());
        std::cout << parser;
        std::exit(0);
    }

    // Verbose output flag
    bool isVerbose = parser.get<bool>("--verbose");

    // Whether use local media file
    bool isLocalFile = parser.get<bool>("--local");

    // Get media file path (local or via rtsp streaming)
    auto file = parser.get("--file");

    // Get resize and rotate args
    auto rotateFlag = parser.get<cv::RotateFlags>("--rotate");
    auto resize = parser.get<cv::Size>("--resize");

    auto outputDir = parser.get("--output");

    int maxNumBlobs = parser.get<int>("--max_blob_count");

    std::unique_ptr<VideoReader> videoReader;

    // Open local media file
    if (isLocalFile) {
        videoReader = std::make_unique<VideoReader>(
            file,
            VideoReaderParams{
                .hardwareAcceleration = "videotoolbox",
                .rotateFlag = rotateFlag,
                .resize = resize,
            });
    } else {
        auto addr = parser.get("--addr");
        auto port = parser.get<uint16_t>("--port");
        auto protocol = parser.get("--rtsp_transport");

        // Get local config
        auto bufferSize = parser.get<size_t>("--buffer_size");

        videoReader =
            std::make_unique<VideoReader>(addr,
                                          file,
                                          port,
                                          VideoReaderParams{
                                              .receiveBufferSize = bufferSize,
                                              .rtspTransport = protocol,
                                              .rotateFlag = rotateFlag,
                                              .resize = resize,
                                          });
    }

    if (!videoReader->isOpened()) {
        if (isVerbose) {
            std::printf("[VIDEO READER] Open failed\n");
        }
        std::exit(EXIT_FAILURE);
    }

    if (isVerbose) {
        std::printf("[VIDEO READER] Successfully opened\n");
    }

    int height = videoReader->getHeight();
    int width = videoReader->getWidth();
    double fps = videoReader->getFPS();

    auto logInterval = parser.get<size_t>("--log_interval");
    logInterval =
        (logInterval == 0) ? static_cast<size_t>(std::round(fps)) : logInterval;

    // Create vibe algorithm instance
    auto vibe = std::make_unique<ViBeSequential>(height, width, 14, 20, 2, 5);
    // Create tracker instance
    auto tracker = std::make_unique<SortTracker>(3, 3);

    // Register callback for tracker
    cv::Mat anno;
    tracker->setTrajectoryEndedCallback(
        [&outputDir, &anno, isVerbose](int tag, const Trajectory& trajectory) {
            // Draw trajectory on annotated image
            trajectory.draw(anno);
            auto timestamp =
                trajectory.getStartTime().time_since_epoch().count();

            // Format trajectory name with tag and timestamp
            std::array<char, 64> str;
            std::sprintf(str.data(),
                         "%s/trajectory_%d_%lld.jpg",
                         outputDir.c_str(),
                         tag,
                         timestamp);

            if (isVerbose) {
                std::printf("[TRAJECTORY] Saved to %s\n", str.data());
            }

#if defined(ROCKCHIP_PLATFORM)
            cv::imwrite(str.data(), anno);
#else
            cv::imshow(str.data(), anno);
            cv::waitKey();
            cv::destroyWindow(str.data());
#endif
        });

    auto detections = std::vector<cv::Rect2f>(8);
    cv::Mat fgMask(height, width, CV_8U);
    cv::Mat updateMask(height, width, CV_8U);
    cv::Mat fgBlobLabels(height, width, CV_32S);
    cv::Mat fgBlobCentroids(64, 2, CV_64F);
    cv::Mat fgBlobStats(64, 5, CV_32S);

    // Prepare structure elements for morphological filtering
    cv::Mat se3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::Mat se5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
    cv::Mat se7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});

    // Prepare runtime measurement
    auto tm = cv::TickMeter();

    // auto colors = Utils::getRandomColors<32>();

    // Start play
    auto frame = cv::Mat(height, width, CV_8UC3);
    while (videoReader->read(frame)) {

#if defined(ROCKCHIP_PLATFORM)
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(16ms);
#else
        if (cv::waitKey(16) == static_cast<int>('q')) {
            if (isVerbose) {
                std::printf("[STOP REQUESTED]\n");
            }
            break;
        }
#endif

        /* Segmentation and update. */
        tm.reset();
        tm.start();

        // Run background segmentation with ViBe
        vibe->segment(frame, fgMask);

        // Process update mask
        cv::morphologyEx(fgMask, updateMask, cv::MORPH_OPEN, se3x3);

        // Update ViBe
        vibe->update(frame, updateMask);

        // Post-processing on foreground mask
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, se3x3);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, se5x5);

        tm.stop();

        double vibeProcessTimeMs = tm.getTimeMilli();

        // Find all connected components
        int numFgBlobs = cv::connectedComponentsWithStats(
            fgMask, fgBlobLabels, fgBlobStats, fgBlobCentroids);

        if (numFgBlobs > maxNumBlobs) {
            // Too many blobs, consider this frame invalid

#if !defined(ROCKCHIP_PLATFORM)
            cv::imshow("frame", frame);
            cv::imshow("fgmask", fgMask);
            cv::imshow("update mask", updateMask);
#endif

            tracker->clear();
            continue;
        }

        detections.clear();
        for (int i = 1; i < numFgBlobs; i++) {
            auto* blobStat = fgBlobStats.ptr<int>(i);

            int x = blobStat[cv::CC_STAT_LEFT] - 6;
            int y = blobStat[cv::CC_STAT_TOP] - 6;
            int w = blobStat[cv::CC_STAT_WIDTH] + 12;
            int h = blobStat[cv::CC_STAT_HEIGHT] + 12;
            // int a = blobStat[cv::CC_STAT_AREA];

            // Add new bbox
            detections.emplace_back(x, y, w, h);

            // auto color = colors.row(i % colors.rows);
            cv::rectangle(frame, {x, y, w, h}, {255, 50, 0}, 1);
        }

        tm.reset();
        tm.start();

        // Update tracker with newly detected bboxes
        tracker->update(detections, frame);

        tm.stop();
        double trackingTimeMs = tm.getTimeMilli();

        std::array<char, 64> str;
        std::sprintf(str.data(),
                     "[PROCESS TIME] ViBe: %.2f ms, Tracking: %.2f",
                     vibeProcessTimeMs,
                     trackingTimeMs);

        if (isVerbose && videoReader->getFrameCount() % logInterval == 0) {
            std::printf("%s\n", str.data());
        }

        // Draw process time measurement result on current frame
        cv::putText(frame,
                    str.data(),
                    {12, 30},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    {0, 0, 255},
                    1,
                    cv::LINE_AA);

// Draw results
#if defined(ROCKCHIP_PLATFORM)
        if (isVerbose && videoReader->getFrameCount() % logInterval == 0) {
            cv::imwrite(outputDir + "/frame.png", frame);
            cv::imwrite(outputDir + "/fgmask.png", fgMask);
            cv::imwrite(outputDir + "/update_mask.png", updateMask);
        }
#else
        cv::imshow("frame", frame);
        cv::imshow("fgmask", fgMask);
        cv::imshow("update mask", updateMask);
#endif
    }

#if !defined(ROCKCHIP_PLATFORM)
    cv::destroyAllWindows();
#endif

    // Done
    return 0;
}