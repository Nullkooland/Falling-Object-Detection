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
// #include "ViBeBase.h"
#include "decoder.hpp"
#include "tracker.hpp"
#include "trajectory.hpp"
#include "utils.hpp"
#include "vibe_sequential.hpp"

#include <argparse/argparse.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#if defined(ROCKCHIP_PLATFORM)
#include <rga/im2d.h>
#include <rga/rga.h>
#else
#include <opencv2/highgui.hpp>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/imgutils.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>

#if defined(ROCKCHIP_PLATFORM)
#include <libavutil/hwcontext_drm.h>
#endif
}

static constexpr size_t MIN_BUFFER_SIZE = 1024 * 64;

namespace fs = std::filesystem;

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
        .default_value(std::string("554"));

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
        .default_value(1024UL * 256)
        .action([](const std::string& arg) { return std::stoul(arg); });

    parser.add_argument("--resize")
        .help("Resize to given height")
        .default_value(-1)
        .action([](const std::string& arg) { return std::stoi(arg); });

    parser.add_argument("--rotate")
        .help("Rotate video by 90 degrees clockwise")
        .default_value(false)
        .implicit_value(true);

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

    // Setup FFMpeg config dictionary
    AVDictionary* avOptions = nullptr;
    // Prepare needed context
    auto* avFormatContext = avformat_alloc_context();

    // Verbose output flag
    bool isVerbose = parser.get<bool>("--verbose");

    // Whether use local media file
    bool isLocalFile = parser.get<bool>("--local");

    // Get media file path (local or via rtsp streaming)
    auto file = parser.get("--file");

    auto outputDir = parser.get("--output");

    int maxNumBlobs = parser.get<int>("--max_blob_count");

    // Open local media file
    if (isLocalFile) {
        if (!fs::is_regular_file(file)) {
            std::fprintf(
                stderr, "[ERROR] Media file %s is invalid\n", file.c_str());
            std::exit(EXIT_FAILURE);
        }

        // Open read the input media file
        if (avformat_open_input(
                &avFormatContext, file.c_str(), nullptr, &avOptions) != 0) {
            std::fprintf(stderr, "[ERROR] Failed to open %s\n", file.c_str());
            std::exit(EXIT_FAILURE);
        } else if (isVerbose) {
            std::printf("[Local file opened] %s\n", file.c_str());
        }
        // Open RTSP stream
    } else {
        auto addr = parser.get("--addr");
        auto port = parser.get("--port");
        auto url = "rtsp://" + addr + ':' + port + '/' + file;
        auto protocol = parser.get("--rtsp_transport");

        // Get local config
        auto bufferSize = parser.get<size_t>("--buffer_size");
        bufferSize = std::min(bufferSize, MIN_BUFFER_SIZE);

        av_dict_set(
            &avOptions, "buffer_size", std::to_string(bufferSize).c_str(), 0);
        av_dict_set(&avOptions, "rtsp_transport", protocol.c_str(), 0);
        av_dict_set(&avOptions,
                    "stimeout",
                    "8000000",
                    0); // Maximum connection time (us)
        av_dict_set(&avOptions, "max_delay", "800000", 0); // Maximum delay (us)

        // Init network
        avformat_network_init();

        // Establish RTSP connection with the server
        if (avformat_open_input(
                &avFormatContext, url.c_str(), nullptr, &avOptions) != 0) {
            std::fprintf(stderr, "[ERROR] Failed to open %s\n", url.c_str());
            std::exit(EXIT_FAILURE);
        } else if (isVerbose) {
            std::printf("[Connection established] %s\n", url.c_str());
        }
    }

    // Read stream info
    if (avformat_find_stream_info(avFormatContext, &avOptions) < 0) {
        std::fprintf(stderr, "[ERROR] Failed to find stream info\n");
        std::exit(EXIT_FAILURE);
    }

    // Find available video stream and decoder
    AVCodec* avDecoder = nullptr;
    // Find video stream in the media file
    int videoStreamIndex = av_find_best_stream(
        avFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &avDecoder, 0);

    if (videoStreamIndex < 0) {
        // Video stream not found, abort
        std::fprintf(stderr, "[ERROR] No video stream found!\n");
        std::exit(EXIT_FAILURE);
    }

    auto* videoStream = avFormatContext->streams[videoStreamIndex];
    auto* decoderParameters = videoStream->codecpar;
    int widthRaw = decoderParameters->width;
    int heightRaw = decoderParameters->height;

    // Extract framerate
    double fps = av_q2d(videoStream->r_frame_rate);
    auto logInterval = parser.get<size_t>("--log_interval");

    logInterval =
        (logInterval == 0) ? static_cast<size_t>(std::round(fps)) : logInterval;

    // Resize along height
    int resizedHeight = parser.get<int>("--resize");

    bool isResize = false;
    int width;
    int height;

    if (resizedHeight > 320 && resizedHeight <= heightRaw) {
        isResize = true;

        height = resizedHeight;
        width = static_cast<int>((static_cast<float>(height) / heightRaw) *
                                 widthRaw);
    } else {
        height = heightRaw;
        width = widthRaw;
    }

    bool isRotate = parser.get<bool>("--rotate");
    if (isRotate) {
        std::swap(width, height);
    }

    // Init decoder
    auto decoder =
        std::unique_ptr<Decoder>(Decoder::create(avDecoder, decoderParameters));

    if (decoder == nullptr) {
        std::fprintf(stderr, "[ERROR] Failed to initialize decoder\n");
        std::exit(EXIT_FAILURE);
    }

    if (isVerbose) {
        std::printf("Found video stream #%d, codec: %d\n",
                    videoStreamIndex,
                    decoder->getCodecID());
    }

    if (!decoder->open(&avOptions)) {
        std::fprintf(stderr, "[ERROR] Failed to open decoder!\n");
        std::exit(EXIT_FAILURE);
    }

    // Allocate frame packet
    AVPacket* avPacket = av_packet_alloc();
    AVFrame* avFrameYUV = av_frame_alloc();

    // auto videoWriter = cv::VideoWriter(
    //     "data/test.mp4", cv::VideoWriter::fourcc('h', '2', '6', '4'), 15,
    //     {1280, 720});

    cv::Mat frame;
    cv::Mat tempFrame;
    size_t frameCount = 0;

#if defined(ROCKCHIP_PLATFORM)
    // Allocate RGB frame
    frame = cv::Mat(height, width, CV_8UC3);

    if (isRotate) {
        tempFrame = cv::Mat(width, height, CV_8UC3);
    }
#else
    // Allocate AVFrame for RGB data
    SwsContext* swsContext = nullptr;
    auto* avFrameBGR = av_frame_alloc();
    avFrameBGR->width = isRotate ? height : width;
    avFrameBGR->height = isRotate ? width : height;

    av_image_alloc(avFrameBGR->data,
                   avFrameBGR->linesize,
                   avFrameBGR->width,
                   avFrameBGR->height,
                   AVPixelFormat::AV_PIX_FMT_BGR24,
                   16);

    // Wrap AVFrame to OpenCV Mat (underlying buffer is shared with AVFrame)
    if (isRotate) {
        frame = cv::Mat(height, width, CV_8UC3);
        tempFrame = cv::Mat(width,
                            height,
                            CV_8UC3,
                            avFrameBGR->data[0],
                            avFrameBGR->linesize[0]);
    } else {
        frame = cv::Mat(height,
                        width,
                        CV_8UC3,
                        avFrameBGR->data[0],
                        avFrameBGR->linesize[0]);
    }

#endif

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
    int64_t tickBegin;
    int64_t tickEnd;
    auto getTimespanMs = [](int64_t tickBegin, int64_t tickEnd) {
        return static_cast<double>(tickEnd - tickBegin) /
               cv::getTickFrequency() * 1e3;
    };

    // auto colors = Utils::getRandomColors<32>();

    // Start play
    for (;;) {

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

        int ret = av_read_frame(avFormatContext, avPacket);
        if (ret == AVERROR_EOF) {
            if (isVerbose) {
                std::printf("[END OF STREAM]\n");
            }
            break;
        }

        if (ret >= 0 && avPacket->stream_index == videoStreamIndex) {
            tickBegin = cv::getTickCount();
            *decoder << avPacket;
            *decoder >> avFrameYUV;

            if (decoder->getStatus() == -EAGAIN) {
                if (isVerbose) {
                    std::printf(
                        "[WAIT PACKET] need more packet to decode frame.\n");
                }
                continue;
            }

            frameCount++;

            if (avFrameYUV->format == -1) {
                if (isVerbose) {
                    std::printf("[DECODE FAILED] #frame %6zu\n", frameCount);
                }
                continue;
            }

            if (isVerbose && frameCount % logInterval == 0) {
                std::printf("[DECODE SUCCESS] #frame %6zu\n", frameCount);
            }

#if defined(ROCKCHIP_PLATFORM)
            // Do YUV420P to RGB conversion
            auto* desc =
                reinterpret_cast<AVDRMFrameDescriptor*>(avFrameYUV->data[0]);
            int fd = desc->objects[0].fd;

            auto src = rga::wrapbuffer_fd_t(fd,
                                            widthRaw,
                                            heightRaw,
                                            widthRaw,
                                            heightRaw,
                                            RK_FORMAT_YCbCr_420_SP);

            auto dst = rga::wrapbuffer_virtualaddr_t(isRotate ? tempFrame.data
                                                              : frame.data,
                                                     isRotate ? height : width,
                                                     isRotate ? width : height,
                                                     isRotate ? height : width,
                                                     isRotate ? width : height,
                                                     RK_FORMAT_BGR_888);

            rga::imcvtcolor_t(src,
                              dst,
                              src.format,
                              dst.format,
                              rga::IM_COLOR_SPACE_DEFAULT,
                              0);

            if (isRotate) {
                src = dst;
                dst = rga::wrapbuffer_virtualaddr_t(frame.data,
                                                    width,
                                                    height,
                                                    width,
                                                    height,
                                                    RK_FORMAT_BGR_888);

                rga::imrotate_t(src, dst, rga::IM_HAL_TRANSFORM_ROT_90, 0);
            }

            rga::imsync();
#else

            swsContext = sws_getCachedContext(
                swsContext,
                avFrameYUV->width,
                avFrameYUV->height,
                static_cast<AVPixelFormat>(avFrameYUV->format),
                avFrameBGR->width,
                avFrameBGR->height,
                AVPixelFormat::AV_PIX_FMT_BGR24,
                SWS_FAST_BILINEAR,
                nullptr,
                nullptr,
                nullptr);

            sws_scale(swsContext,
                      avFrameYUV->data,
                      avFrameYUV->linesize,
                      0,
                      avFrameYUV->height,
                      avFrameBGR->data,
                      avFrameBGR->linesize);

            if (isRotate) {
                cv::rotate(tempFrame, frame, cv::ROTATE_90_CLOCKWISE);
            }
#endif
            tickEnd = cv::getTickCount();
            double decodeTimeMs = getTimespanMs(tickBegin, tickEnd);

            /* Segmentation and update. */
            tickBegin = cv::getTickCount();

            vibe->segment(frame, fgMask);

            cv::morphologyEx(fgMask, updateMask, cv::MORPH_OPEN, se3x3);

            // Update ViBe
            vibe->update(frame, updateMask);

            // Post-processing on foreground mask
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, se3x3);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, se5x5);

            tickEnd = cv::getTickCount();

            double vibeProcessTimeMs = getTimespanMs(tickBegin, tickEnd);

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

            tickBegin = cv::getTickCount();

            // Update tracker with newly detected bboxes
            tracker->update(detections, frame);

            tickEnd = cv::getTickCount();
            double trackingTimeMs = getTimespanMs(tickBegin, tickEnd);

            std::array<char, 64> str;
            std::sprintf(
                str.data(),
                "[PROCESS TIME] Decode: %.2f ms, ViBe: %.2f ms, Tracking: %.2f",
                decodeTimeMs,
                vibeProcessTimeMs,
                trackingTimeMs);

            if (isVerbose && frameCount % logInterval == 0) {
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
            if (isVerbose && frameCount % logInterval == 0) {
                cv::imwrite(outputDir + "/frame.png", frame);
                cv::imwrite(outputDir + "/fgmask.png", fgMask);
                cv::imwrite(outputDir + "/update_mask.png", updateMask);
            }
#else
            cv::imshow("frame", frame);
            cv::imshow("fgmask", fgMask);
            cv::imshow("update mask", updateMask);
            // videoWriter << matFrame;
#endif
        }
    }

    // Clean things up
    av_dict_free(&avOptions);
    av_packet_free(&avPacket);
    av_frame_free(&avFrameYUV);
#if !defined(ROCKCHIP_PLATFORM)
    av_frame_free(&avFrameBGR);
#endif

    avformat_free_context(avFormatContext);

    if (!isLocalFile) {
        avformat_network_deinit();
    }

#if !defined(ROCKCHIP_PLATFORM)
    cv::destroyAllWindows();
    // videoWriter.release();
#endif

    // Done
    return 0;
}