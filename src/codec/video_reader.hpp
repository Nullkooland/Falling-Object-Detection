#pragma once

#include <opencv2/core.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/imgutils.h>
#include <libavutil/rational.h>
#if !defined(ROCKCHIP_PLATFORM)
#include <libswscale/swscale.h>
#endif
}

/**
 * @brief Additional parameters for VideoReader class
 */
struct VideoReaderParams {
    /**
     * @brief RTSP receive buffer size (bytes)
     */
    size_t receiveBufferSize = 1024 * 256;

    /**
     * @brief Connection timeout (us)
     */
    int connectionTimeoutUs = 8000000;

    /**
     * @brief Maximum delay (us)
     */
    int maxDelayUs = 8000000;

    /**
     * @brief RTSP transport protocol
     */
    std::string_view rtspTransport = "tcp";

    /**
     * @brief Rotation flag (cv::RotateFlag)
     */
    int rotateFlag = -1;

    /**
     * @brief Resize (output size)
     */
    cv::Size resize = {0, 0};
};

/**
 * @brief Video reader class
 */
class VideoReader {
  public:
#pragma region Public member methods
    /**
     * @brief Construct a new VideoReader object that reads remote RTSP stream
     *
     * @param addr Remote address
     * @param filename Video stream file name
     * @param port Port number
     * @param params Additional parameters
     */
    VideoReader(const std::string& addr,
                const std::string& filename,
                uint16_t port = 554,
                const VideoReaderParams& params = VideoReaderParams());

    /**
     * @brief Construct a new Video Reader object that reads local video file
     *
     * @param filename Local video file
     * @param params Additional parameters
     */
    VideoReader(std::string_view filename,
                const VideoReaderParams& params = VideoReaderParams());

    /**
     * @brief Destroy the VideoReader object
     *
     */
    ~VideoReader();

    /**
     * @brief Read a frame
     *
     * @param frame Output frame
     * @return true Read successfully
     * @return false Read failed
     */
    bool read(cv::Mat& frame);

    /**
     * @brief Read a frame
     *
     * @param frame Output frame
     * @return true Read successfully
     * @return false Read failed
     */
    bool operator>>(cv::Mat& frame) { return read(frame); }

    /**
     * @brief Close the video stream and free all internal data
     *
     */
    void close();

    /**
     * @brief Tells whether this VideoReader has successfully opened a stream
     *
     * @return true This VideoReader is opened and ready to read
     * @return false This VideoReader is not opened
     */
    bool isOpened() const { return _isOpened; }

    int getWidth() const { return _width; }

    int getHeight() const { return _height; }

    double getFPS() const { return _fps; }

    int getFrameCount() const { return _frameCount; }

#pragma endregion

  private:
#pragma region Private constants

    /**
     * @brief Maximum number of errors occurred before reporting failure when
     * reading a frame
     *
     */
    static constexpr int MAX_NUM_ERRORS = 500;

#pragma endregion

#pragma region Private member variables

    AVFormatContext* _avFormatContext;
    AVDictionary* _avFormatOptions;

    AVCodecContext* _avDecoderContext;

    AVPacket* _avPacket;
    AVFrame* _avFrameRaw;
    AVFrame* _avFrameBGR24;

#if !defined(ROCKCHIP_PLATFORM)
    SwsContext* _swsContext;
#endif

    bool _isOpened;

    int _heightRaw;
    int _widthRaw;
    int _height;
    int _width;
    double _fps;
    int _frameCount;

    int _streamIndex;
    int _rotateFlag;

#pragma endregion

#pragma region Private member methods

    /**
     * @brief Default constructor
     * 
     */
    VideoReader();

    /**
     * @brief Open a stream after initialization
     * 
     * @param path Stream path (local or remote)
     * @param params Additional parameters
     */
    void open(std::string_view path, const VideoReaderParams& params);

    /**
     * @brief Apply post processing to the decoded raw frame
     * 
     * @param frame Raw frame
     * @return true Successfully applied post-processing
     * @return false Failed to apply post-processing
     */
    bool postProcess(cv::Mat& frame);

#pragma endregion
};