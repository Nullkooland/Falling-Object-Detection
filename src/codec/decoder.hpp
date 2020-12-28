/**
 * @file decoder.cpp
 * @author Xiaoran Weng (goose_bomb@outlook.com)
 * @brief Wrapper of FFMpeg decoder
 * @version 0.1
 * @date 2020-12-28
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

extern "C" {
#include "libavcodec/avcodec.h"
}

class Decoder {
  public:
    static Decoder* create(const AVCodec* avDecoder,
                           const AVCodecParameters* parameters);

    AVCodecID getCodecID() const { return _context->codec_id; }

    Decoder(AVCodecContext* context);

    Decoder() = delete;

    ~Decoder();

    bool open(AVDictionary** options);

    Decoder& operator<<(const AVPacket* packet);

    Decoder& operator>>(AVFrame* frame);

    operator bool() const;

    int getStatus() const { return _status; }

  private:
    bool _isDecodingOK;
    int _status;
    AVCodecContext* _context;

    static bool isCodecSupported(AVCodecID id) {
        return (id == AV_CODEC_ID_H264) || (id == AV_CODEC_ID_H265) ||
               (id == AV_CODEC_ID_VP8) || (id == AV_CODEC_ID_VP9);
    }

    Decoder(AVCodecID id, AVCodec* codec, AVCodecContext* context);
};