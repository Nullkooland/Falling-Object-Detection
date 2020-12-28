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
#include "decoder.hpp"

#include <cstdio>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>

Decoder* Decoder::create(const AVCodec* avDecoder,
                         const AVCodecParameters* parameters) {
    if (!isCodecSupported(avDecoder->id)) {
        std::fprintf(
            stderr, "[ERROR] [DECODER] Unsupported codec: %d\n", avDecoder->id);
        return nullptr;
    }

    AVCodecContext* context = avcodec_alloc_context3(avDecoder);

    if (context == nullptr) {
        std::fprintf(stderr, "[ERROR] [DECODER] Failed to find ffmpeg decoder");
        return nullptr;
    }

    if (avcodec_parameters_to_context(context, parameters) < 0) {
        std::fprintf(stderr, "[ERROR] [DECODER] Failed to initialize decoder");
        return nullptr;
    }

    return new Decoder(context);
}

Decoder::Decoder(AVCodecContext* context) : _context{context} {}

Decoder::~Decoder() { avcodec_free_context(&_context); }

bool Decoder::open(AVDictionary** options = nullptr) {
    if (avcodec_open2(_context, _context->codec, options) != 0) {
        return false;
    }

    _isDecodingOK = true;
    return true;
}

Decoder& Decoder::operator<<(const AVPacket* packet) {
    _status = avcodec_send_packet(_context, packet);
    if (_status != 0) {
        _isDecodingOK = false;
    }
    return *this;
}

Decoder& Decoder::operator>>(AVFrame* frame) {
    _status = avcodec_receive_frame(_context, frame);

    if (_status != 0) {
        _isDecodingOK = false;
    }
    return *this;
}

Decoder::operator bool() const { return _isDecodingOK; }