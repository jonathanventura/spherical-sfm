#pragma once

#include <Eigen/Core>

#include <opencv2/core.hpp>

#include <sphericalsfm/sfm_types.h>

namespace stereopanotools {

    struct Keyframe
    {
        int index;
        
        Eigen::Vector3d t;
        Eigen::Vector3d r;
        Eigen::Matrix3d R;
        
        Eigen::Matrix3d Ry;
        Eigen::Matrix3d Rxz; // R = Rxz * Ry
        
        double theta;
        
        cv::Mat image;
        cv::Mat image_gray;
        cv::Mat image_float;
        cv::Mat half_image_gray;
        cv::Mat mask;
        
        cv::Mat forward_flow;
        cv::Mat backward_flow;
    };

    void make_stereo_panoramas( const sphericalsfm::Intrinsics &intrinsics, const std::string &videopath, const std::string &outputpath,
        const int panowidth, const bool is_loop ); 

}

