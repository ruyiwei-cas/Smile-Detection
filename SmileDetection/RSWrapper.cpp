/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#include <iostream>
#include <opencv2/contrib/contrib.hpp>
#include <librealsense/rs.hpp>
#include "RSWrapper.h"

using namespace std;

class RSWrapper::Impl
{
public:
    Impl(int CImgSize = 1, int DImgSize = 1, int Cfps = 30, int Dfps = 30, int depth_preset = 0);
    ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    int init();
    void release();
    bool capture(int idx, cv::Mat &color, cv::Mat &depth, bool smooth = false);
    cv::Mat visualDepth(cv::Mat &depth);

    int innerBandThreshold;
    int outerBandThreshold;

private:
    cv::Mat get_frame_data(int idx, rs::stream stream);
    cv::Mat smoothing(cv::Mat &depth, unsigned short **arr = nullptr, int N = 0, int space = 1, int time = 0);

    rs::context *ctx;
    vector<rs::device *> devices;
    int c_width, c_height;
    int d_width, d_height;
    int c_fps;
    int d_fps;
    int depth_preset;
    cv::Mat visual_dimg;

    cv::Mat smoothed;
};

RSWrapper::RSWrapper(int CImgSize, int DImgSize, int Cfps, int Dfps, int depth_preset)
        : impl(new Impl(CImgSize, DImgSize, Cfps, Dfps, depth_preset)) {}
RSWrapper::~RSWrapper() {}

int RSWrapper::init()
{
    return impl->init();
}

void RSWrapper::release()
{
    impl->release();
}

bool RSWrapper::capture(int idx, cv::Mat &color, cv::Mat &depth, bool smooth)
{
    return impl->capture(idx, color, depth, smooth);
}

cv::Mat RSWrapper::visualDepth(cv::Mat &depth)
{
    return impl->visualDepth(depth);
}

void RSWrapper::setSmoothConfig(int innerBandThreshold, int outerBandThreshold)
{
    impl->innerBandThreshold = innerBandThreshold;
    impl->outerBandThreshold = outerBandThreshold;
}

RSWrapper::Impl::Impl(int CImgSize, int DImgSize, int Cfps, int Dfps, int depth_preset)
    : innerBandThreshold(4), outerBandThreshold(10)
{
    switch (CImgSize) {
        case 0:
            c_width = 320;
            c_height = 240;
            break;
        case 1:
        default:
            c_width = 640;
            c_height = 480;
            break;
        case 2:
            c_width = 1920;
            c_height = 1080;
            break;
    }

    switch (DImgSize) {
        case 0:
            d_width = 320;
            d_height = 240;
            break;
        case 1:
        default:
            d_width = 480;
            d_height = 360;
            break;
        case 2:
            d_width = 628;
            d_height = 468;
            break;
    }

    if (Cfps == 15 || Cfps == 30 || Cfps == 60)
        c_fps = Cfps;
    else
        c_fps = 30;
    if (Dfps == 30 || Dfps == 60 || Dfps == 90)
        d_fps = Dfps;
    else
        d_fps = 30;

    if (depth_preset >= 0 && depth_preset <= 5)
        this->depth_preset = depth_preset;
    else
        this->depth_preset = 0;
}

int RSWrapper::Impl::init()
{
    int idx = 0;
    try {
        ctx = new rs::context();
        idx = ctx->get_device_count();
        for (auto i = 0; i < idx; i++)
            devices.push_back(ctx->get_device(i));

        for (auto dev : devices) {
            dev->enable_stream(rs::stream::depth, d_width, d_height, rs::format::z16, d_fps);
            dev->enable_stream(rs::stream::color, c_width, c_height, rs::format::bgr8, c_fps);
            rs::apply_depth_control_preset(dev, depth_preset);
            dev->start();
        }
    } catch (const rs::error &e) {
        std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    return idx;
}

void RSWrapper::Impl::release()
{
    try {
        for (auto dev : devices)
            dev->stop();
        devices.clear();
        delete ctx;
        ctx = nullptr;
    } catch (const rs::error &e) {
        std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

bool RSWrapper::Impl::capture(int idx, cv::Mat &color, cv::Mat &depth, bool smooth)
{
    if (idx >= devices.size() || idx < 0) {
        std::cerr << "RSWrapper: wrong device idx: " << idx << std::endl;
        return false;
    }

    auto dev = devices[idx];
    try {
        dev->wait_for_frames();
        color = get_frame_data(idx, rs::stream::rectified_color);
        depth = get_frame_data(idx, rs::stream::depth_aligned_to_rectified_color);
        if (color.empty() || depth.empty())
            return false;
        if (smooth)
            depth = smoothing(depth);
    } catch (const rs::error &e) {
        std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
        return false;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    return true;
}

cv::Mat RSWrapper::Impl::visualDepth(cv::Mat &depth)
{
    depth.convertTo(visual_dimg, CV_8UC1, 1.0/16);
    cv::applyColorMap(visual_dimg, visual_dimg, cv::COLORMAP_RAINBOW);
    return visual_dimg;
}

cv::Mat RSWrapper::Impl::get_frame_data(int idx, rs::stream stream)
{
    auto dev = devices[idx];
    assert(dev->is_stream_enabled(stream));

    int w = dev->get_stream_width(stream);
    int h = dev->get_stream_height(stream);
    const void *data = dev->get_frame_data(stream);
    rs::format format = dev->get_stream_format(stream);

    switch (format) {
        case rs::format::z16:
            return cv::Mat(cv::Size(w, h), CV_16UC1, const_cast<void *>(data), cv::Mat::AUTO_STEP);
        case rs::format::bgr8:
            return cv::Mat(cv::Size(w, h), CV_8UC3, const_cast<void *>(data), cv::Mat::AUTO_STEP);
        default:
            break;
    }

    return cv::Mat();
}

cv::Mat RSWrapper::Impl::smoothing(cv::Mat &depth, unsigned short **arr, int N, int space, int time)
{
    int nl = depth.rows;
    int nc = depth.cols;
    int nc_cols = depth.cols;
    smoothed = depth.clone();

    for (int j = 0; j < nl; j++) {
        unsigned short *src = depth.ptr<unsigned short>(j);
        unsigned short *dst = smoothed.ptr<unsigned short>(j);
        for (int i = 0; i < nc; i++) {
            // fix wrong data
            if (*src < 300) {
                int filterCollection[24][2] = {0};
                int innerBandCount = 0;
                int outerBandCount = 0;
                for (int subi = -2; subi <= 2; ++subi) {
                    for (int subj = -2; subj <= 2; ++subj) {
                        if (i + subi >= 0 && i + subi < nc && subj * nc_cols + subi >= 0 && j + subj < nl) {
                            double valueTemp = *(src + subj * nc_cols + subi);
                            if (valueTemp > 300) {
                                for (int ii = 0; ii < 24; ii++) {
                                    if (filterCollection[ii][0] == valueTemp) {
                                        filterCollection[ii][1]++;
                                    } else if (filterCollection[ii][0] == 0) {
                                        filterCollection[ii][0] = (int)valueTemp;
                                        filterCollection[ii][1]++;
                                    }
                                    if (subi != 2 && subj != -2 && subi != 2 && subj != -2)
                                        innerBandCount++;
                                    else
                                        outerBandCount++;
                                }
                            }
                        }
                    }
                }

                if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold) {
                    int frequency = 0;
                    int value = 0;
                    for (int i = 0; i < 24; i++) {
                        if (filterCollection[i][0] == 0)
                            break;
                        if (filterCollection[i][1] > frequency) {
                            value = filterCollection[i][0];
                            frequency = filterCollection[i][1];
                        }
                    }
                    if (space == 1) {
                        *dst = value;
                    } else
                        *dst = *src;
                } else {
                    double PastValue = 0;
                    double PaseSum = 1;
                    for (int qv = 0; qv < N; ++qv) {
                        unsigned short * tempPtr = arr[qv];
                        double tempPast = *(tempPtr + i + j * nc);
                        if (tempPast > 300) {
                            PaseSum = PaseSum + qv;
                            PastValue = PastValue + tempPast *(qv);
                        }
                    }
                    PastValue = PastValue / PaseSum;

                    if (PastValue > 300 && time == 1) {
                        *dst = (unsigned short)PastValue;
                    } else {
                        *dst = *src;
                    }
                }

                src++;
                dst++;
            } else {
                *dst = *src;
                dst++;
                src++;
            }
        }
    }

    return smoothed;
}
