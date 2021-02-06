#include "video_reader.hpp"

#include <cmath>
#include <opencv2/core.hpp>

#if defined(ROCKCHIP_PLATFORM)
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui.hpp>
#endif

constexpr std::string_view videoFilename =
#if defined(ROCKCHIP_PLATFORM)
    "/home/goose_bomb/cv/videos/apartment.264";
#else
    "data/apartment.264";
#endif

int main(int argc, char* argv[]) {
    // auto reader = VideoReader(
    //     videoFilename,
    //     VideoReaderParams{.hardwareAcceleration = "videotoolbox",
    //                       .rotateFlag = cv::RotateFlags::ROTATE_90_CLOCKWISE,
    //                       .resize = {720, 1280}});

    auto reader =
        VideoReader("localhost",
                    "live",
                    554,
                    VideoReaderParams{.hardwareAcceleration = "videotoolbox",
                                      .rtspTransport = "tcp",
                                      .resize = {0, 0}});

    int fps = static_cast<int>(std::round(reader.getFPS()));

    cv::Mat frame;
    cv::TickMeter tm;

    while (true) {
        tm.reset();
        tm.start();

        if (!reader.read(frame)) {
            break;
        }
        tm.stop();

#if !defined(ROCKCHIP_PLATFORM)
        cv::imshow("Frame", frame);
        if (cv::waitKey(16) == static_cast<int>('q')) {
            break;
        }
#endif

        if (reader.getFrameCount() % fps == 0) {
            std::printf("[FRAME READ] #%3d, time: %.2f ms\n",
                        reader.getFrameCount(),
                        tm.getTimeMilli());

#if defined(ROCKCHIP_PLATFORM)
            cv::imwrite("data/out.jpg", frame);
#endif
        }
    }

    reader.close();

#if !defined(ROCKCHIP_PLATFORM)
    cv::destroyAllWindows();
#endif
    return 0;
}