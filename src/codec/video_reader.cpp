#include "video_reader.hpp"

#if defined(ROCKCHIP_PLATFORM)

extern "C" {
#include <drm/drm_fourcc.h>
#include <libavutil/hwcontext_drm.h>
#include <rga/im2d.h>
#include <rga/rga.h>
}

#endif

VideoReader::VideoReader()
    : _avFormatContext(nullptr),
      _avFormatOptions(nullptr),
      _avDecoderContext(nullptr),
      _avPacket(nullptr),
      _avFrameRaw(nullptr),
      _avFrameBGR24(nullptr),
      _frameCount(0) {
    // Allocate necessary objects
    _avFormatContext = avformat_alloc_context();
    _avPacket = av_packet_alloc();
    _avFrameRaw = av_frame_alloc();

#if !defined(ROCKCHIP_PLATFORM)
    _swsContext = nullptr;
#endif
}

VideoReader::VideoReader(const std::string& addr,
                         const std::string& filename,
                         uint16_t port,
                         const VideoReaderParams& params)
    : VideoReader() {

    av_dict_set(&_avFormatOptions,
                "buffer_size",
                std::to_string(params.receiveBufferSize).c_str(),
                0);

    av_dict_set(
        &_avFormatOptions, "rtsp_transport", params.rtspTransport.data(), 0);

    av_dict_set(&_avFormatOptions,
                "stimeout",
                std::to_string(params.connectionTimeoutUs).c_str(),
                0);

    av_dict_set(&_avFormatOptions,
                "max_delay",
                std::to_string(params.maxDelayUs).c_str(),
                0);

    avformat_network_init();

    auto url = "rtsp://" + addr + ':' + std::to_string(port) + '/' + filename;
    open(url, params);
}

VideoReader::VideoReader(std::string_view filename,
                         const VideoReaderParams& params)
    : VideoReader() {
    open(filename, params);
}

void VideoReader::open(std::string_view path, const VideoReaderParams& params) {
    int err;
    // Open stream
    err = avformat_open_input(
        &_avFormatContext, path.data(), nullptr, &_avFormatOptions);
    if (err < 0) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to open path: %s, error: %d\n",
               path.data(),
               err);
        return;
    }

    // Read stream info
    err = avformat_find_stream_info(_avFormatContext, &_avFormatOptions);
    if (err < 0) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to find stream info, error: %d\n",
               err);
        return;
    }

    // Find available video stream and decoder
    AVCodec* avDecoder = nullptr;
    // Find video stream in the media file
    int _streamIndex = av_find_best_stream(
        _avFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &avDecoder, 0);

    if (_streamIndex < 0) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to find valid video stream , error: %d\n",
               err);
        return;
    }

    // Extract info
    const auto* stream = _avFormatContext->streams[_streamIndex];
    _heightRaw = stream->codecpar->height;
    _widthRaw = stream->codecpar->width;
    _fps = av_q2d(stream->r_frame_rate);
    _rotateFlag = params.rotateFlag;

    // Set output size
    if (!params.resize.empty()) {
        _height = params.resize.height;
        _width = params.resize.width;
    } else if (_rotateFlag == cv::RotateFlags::ROTATE_90_CLOCKWISE ||
               _rotateFlag == cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE) {
        _height = _widthRaw;
        _width = _heightRaw;
    } else {
        _height = _heightRaw;
        _width = _widthRaw;
    }

    // Allocate BGR24 frame buffer for intermediate processing
    _avFrameBGR24 = av_frame_alloc();
    if (_rotateFlag == -1 || _rotateFlag == cv::RotateFlags::ROTATE_180) {
        _avFrameBGR24->height = _height;
        _avFrameBGR24->width = _width;
    } else if (_rotateFlag == cv::RotateFlags::ROTATE_90_CLOCKWISE ||
               _rotateFlag == cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE) {
        _avFrameBGR24->height = _width;
        _avFrameBGR24->width = _height;
    }

    err = av_image_alloc(_avFrameBGR24->data,
                         _avFrameBGR24->linesize,
                         _avFrameBGR24->width,
                         _avFrameBGR24->height,
                         AVPixelFormat::AV_PIX_FMT_BGR24,
                         16);

    if (err < 0) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to allocate BGR24 frame buffer, error: %d\n",
               err);
        return;
    }

    // Set up decoder
    _avDecoderContext = avcodec_alloc_context3(avDecoder);

    // Enable multi-threaded decoding
    _avDecoderContext->thread_count = cv::getNumberOfCPUs();
    _avDecoderContext->thread_type = FF_THREAD_FRAME;

    err = avcodec_parameters_to_context(_avDecoderContext, stream->codecpar);
    if (err < 0) {
        av_log(
            nullptr, AV_LOG_ERROR, "Failed to setup decoder, error: %d\n", err);
        return;
    }

    // Open decoder
    err = avcodec_open2(_avDecoderContext, avDecoder, nullptr);
    if (err < 0) {
        av_log(
            nullptr, AV_LOG_ERROR, "Failed to open decoder, error: %d\n", err);
        return;
    }

    _isOpened = true;
}

VideoReader::~VideoReader() { close(); }

bool VideoReader::read(cv::Mat& frame) {
    if (!_isOpened) {
        return false;
    }

    int err;
    int numErrors = 0;
    bool hasGotFrame = false;

    while (!hasGotFrame) {
        err = av_read_frame(_avFormatContext, _avPacket);

        // Wait for more data
        if (err == AVERROR(EAGAIN)) {
            continue;
        }

        // End of stream
        if (err == AVERROR_EOF) {
            return false;
        }

        if (numErrors > MAX_NUM_ERRORS) {
            av_log(nullptr,
                   AV_LOG_ERROR,
                   "Maximum number of errors reached while trying to read a "
                   "frame\n");
            break;
        }

        // Send the packet to decoder
        err = avcodec_send_packet(_avDecoderContext, _avPacket);

        // if (err == AVERROR(EAGAIN)) {
        //     continue;
        // }

        if (err < 0) {
            av_log(nullptr,
                   AV_LOG_ERROR,
                   "Failed to send packet to decoder, error: %d\n",
                   err);
            numErrors++;
            continue;
        }

        // Receive frame from decoder
        err = avcodec_receive_frame(_avDecoderContext, _avFrameRaw);

        // Wait for more packets to decode a frame
        if (err == AVERROR(EAGAIN)) {
            continue;
        }

        // End of stream
        if (err == AVERROR_EOF) {
            return false;
        }

        if (err < 0) {
            av_log(nullptr,
                   AV_LOG_ERROR,
                   "Failed to receive frame from decoder, error: %d\n",
                   err);
            numErrors++;
            continue;
        }

        hasGotFrame = true;
        _frameCount++;
    }

    // Unref the packet
    av_packet_unref(_avPacket);
    return postProcess(frame);
}

bool VideoReader::postProcess(cv::Mat& frame) {
    int err;
#if defined(ROCKCHIP_PLATFORM)
    // Do YUV420P to RGB conversion

    // Extract the DMA buffer that holds the decoded YUV420P frame
    auto* desc = reinterpret_cast<AVDRMFrameDescriptor*>(_avFrameRaw->data[0]);
    int fd = desc->objects[0].fd;

    int drmFormat = desc->layers[0].format;
    int rkFormat;
    if (drmFormat == DRM_FORMAT_NV12) {
        rkFormat = RK_FORMAT_YCbCr_420_SP;
    } else if (drmFormat == DRM_FORMAT_P010) {
        rkFormat = RK_FORMAT_YCbCr_420_SP_10B;
    } else {
        av_log(
            nullptr, AV_LOG_ERROR, "Unsupported frame format: %d\n", drmFormat);
        return false;
    }

    // Wrap YUV frame to src
    auto src = rga::wrapbuffer_fd_t(
        fd, _widthRaw, _heightRaw, _widthRaw, _heightRaw, rkFormat);

    // WRAP BGR24 buffer to dst
    auto dst = rga::wrapbuffer_virtualaddr_t(_avFrameBGR24->data[0],
                                             _avFrameBGR24->width,
                                             _avFrameBGR24->height,
                                             _avFrameBGR24->linesize[0] / 3,
                                             _avFrameBGR24->height,
                                             RK_FORMAT_BGR_888);

    // Convert color space & resize
    err = rga::imcvtcolor_t(
        src, dst, src.format, dst.format, rga::IM_COLOR_SPACE_DEFAULT, 0);

    if (err != rga::IM_STATUS_SUCCESS) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to convert color and resize using RGA, error: %d\n",
               err);
        return false;
    }

    if (_rotateFlag != -1) {
        // Allocate frame if size or type is not matched
        if (frame.rows != _height || frame.cols != _width ||
            frame.type() != CV_8UC3) {
            frame = cv::Mat(_height, _width, CV_8UC3);
        }

        src = dst;
        dst = rga::wrapbuffer_virtualaddr_t(frame.data,
                                            _width,
                                            _height,
                                            frame.step / 3,
                                            _height,
                                            RK_FORMAT_BGR_888);

        err = rga::imrotate_t(src, dst, 1 << _rotateFlag, 0);

        if (err != rga::IM_STATUS_SUCCESS) {
            av_log(nullptr,
                   AV_LOG_ERROR,
                   "Failed to rotate frame using RGA, error: %d\n",
                   err);
            return false;
        }
    } else {
        // Wrap the BGR24 buffer to output cv::Mat
        frame = cv::Mat(_avFrameBGR24->height,
                        _avFrameBGR24->width,
                        CV_8UC3,
                        _avFrameBGR24->data[0],
                        _avFrameBGR24->linesize[0]);
    }

    // Wait for RGA processing to complete
    err = rga::imsync();

    if (err != rga::IM_STATUS_SUCCESS) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to sync RGA, error: %d\n", err);
        return false;
    }

#else
    // Get cached SwsContext object (do not free the object afterwards)
    _swsContext =
        sws_getCachedContext(_swsContext,
                             _widthRaw,
                             _heightRaw,
                             static_cast<AVPixelFormat>(_avFrameRaw->format),
                             _avFrameBGR24->width,
                             _avFrameBGR24->height,
                             AVPixelFormat::AV_PIX_FMT_BGR24,
                             SWS_FAST_BILINEAR,
                             nullptr,
                             nullptr,
                             nullptr);

    if (_swsContext == nullptr) {
        av_log(nullptr, AV_LOG_ERROR, "Failed to get SwsContext object\n");
        return false;
    }

    // Convert color space & resize
    err = sws_scale(_swsContext,
                    _avFrameRaw->data,
                    _avFrameRaw->linesize,
                    0,
                    _avFrameRaw->height,
                    _avFrameBGR24->data,
                    _avFrameBGR24->linesize);

    if (err < 0) {
        av_log(nullptr,
               AV_LOG_ERROR,
               "Failed to convert color and resize, error: %d\n",
               err);
        return false;
    }

    if (_rotateFlag != -1) {
        // Allocate frame if size or type is not matched
        if (frame.rows != _height || frame.cols != _width ||
            frame.type() != CV_8UC3) {
            frame = cv::Mat(_height, _width, CV_8UC3);
        }

        // Wrap the BGR24 buffer to temp cv::Mat
        auto temp = cv::Mat(_avFrameBGR24->height,
                            _avFrameBGR24->width,
                            CV_8UC3,
                            _avFrameBGR24->data[0],
                            _avFrameBGR24->linesize[0]);

        // Rotate the temp frame to output frame
        cv::rotate(temp, frame, _rotateFlag);
    } else {
        // Wrap the BGR24 buffer to output cv::Mat
        frame = cv::Mat(_avFrameBGR24->height,
                        _avFrameBGR24->width,
                        CV_8UC3,
                        _avFrameBGR24->data[0],
                        _avFrameBGR24->linesize[0]);
    }
#endif
    // Done
    return true;
}

void VideoReader::close() {
    avformat_network_deinit();

    av_packet_free(&_avPacket);
    av_frame_free(&_avFrameRaw);
    av_frame_free(&_avFrameBGR24);

    av_dict_free(&_avFormatOptions);
    avformat_close_input(&_avFormatContext);
    avformat_free_context(_avFormatContext);
    avcodec_free_context(&_avDecoderContext);

    _isOpened = false;
}