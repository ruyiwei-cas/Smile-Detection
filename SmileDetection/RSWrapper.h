/*
 * License: 3-clause BSD. See LICENSE file in root directory.
 * Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
 * */

#ifndef _RSWRAPPER_H_
#define _RSWRAPPER_H_

#include <memory>
#include <opencv2/core/core.hpp>

#ifdef WIN32

#pragma warning (disable : 4251)

#ifdef RSWrapper_SHARED

#if defined(RSWrapper_EXPORTS)
# define RSAPI __declspec(dllexport)
#else
# define RSAPI __declspec(dllimport)
#endif // RSWRAPPER_EXPORTS

#else

#define RSAPI

#endif // SHARED

#else

#define RSAPI

#endif // WIN32


class RSAPI RSWrapper final
{
public:

    /*
     * Init RealSense wrapper.
     * CImgSize: Color stream size
     *           0 - 320x240
     *           1 - 640x480
     *           2 - 1920x1080
     * DImgSize: Depth stream size
     *           0 - 320x240
     *           1 - 480x360
     *           2 - 628x468
     * Cfps: Color stream fps
     * Dfps: Depth stream fps
     * depth_preset: librealsense builtin preset depth sampling confidence
     */
    RSWrapper(int CImgSize = 1, int DImgSize = 1, int Cfps = 60, int Dfps = 60, int depth_preset = 0);

    ~RSWrapper();
    RSWrapper(const RSWrapper &) = delete;
    RSWrapper &operator=(const RSWrapper &) = delete;

    /*
     * Init and start RealSense devices, return the numbers of connected devices.
     */
    int init();

    /*
     * cleanup RealSense devices.
     */
    void release();

    /*
     * Capture Color and Depth frames from RealSense devices, idx to choose which one.
     * Smooth could be used to smoothing Depth frame.
     * Color frame is CV_8UC3, BGR as OpenCV normally uses.
     * Depth frame is CV_16UC1, aligned to RGB frame size.
     */
    bool capture(int idx, cv::Mat &color, cv::Mat &depth, bool smooth = false);

    /*
     * Set internal smooth parameters.
     */
    void setSmoothConfig(int innerBandThreshold, int outerBanThreshold);

    /*
     * Visualize Depth frame to CV_8UC3 BGR for debug usage.
     */
    cv::Mat visualDepth(cv::Mat &depth);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

#endif
