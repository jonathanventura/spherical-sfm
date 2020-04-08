
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <gflags/gflags.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

DEFINE_string(intrinsics, "distorted.txt", "Path to input intrinsics (focalx focaly centerx centery k1 k2 p1 p2 k3 k4 k5 k6)");
DEFINE_string(intrinsicsout, "intrinsics.txt", "Path to output intrinsics (focal centerx centery)");
DEFINE_int32(rotate, 0, "Rotation to apply to input video (0 for none, 1 for 90 degrees CW, 2 for 180, 3 for 270");
DEFINE_double(focal, 0, "Output focal length (defaults to (focalx+focaly)/2)");
DEFINE_string(video, "%06d.png", "Path to video or image search pattern like %06d.png");
DEFINE_string(output, "output", "Path to output directory for undistorted frames (must exist)");

int main( int argc, char **argv )
{
    srand(0);

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    double focalx, focaly, centerx, centery, k1, k2, p1, p2, k3, k4, k5, k6;
    std::ifstream intrinsicsf( FLAGS_intrinsics );
    intrinsicsf >> focalx >> focaly >> centerx >> centery >> k1 >> k2 >> p1 >> p2 >> k3 >> k4 >> k5 >> k6;

    cv::Mat cameraMatrix = cv::Mat::eye(3,3,CV_64F);
    cameraMatrix.at<double>(0,0) = focalx;
    cameraMatrix.at<double>(0,2) = centerx;
    cameraMatrix.at<double>(1,1) = focaly;
    cameraMatrix.at<double>(1,2) = centery;

    cv::Mat distCoeffs(8,1,CV_64F);
    distCoeffs.at<double>(0,0) = k1;
    distCoeffs.at<double>(1,0) = k2;
    distCoeffs.at<double>(2,0) = p1;
    distCoeffs.at<double>(3,0) = p2;
    distCoeffs.at<double>(4,0) = k3;
    distCoeffs.at<double>(5,0) = k4;
    distCoeffs.at<double>(6,0) = k5;
    distCoeffs.at<double>(7,0) = k6;
    
    double focal = FLAGS_focal;
    if ( FLAGS_focal == 0 ) focal = (focalx+focaly)/2;
    
    cv::Mat newCameraMatrix = cv::Mat::eye(3,3,CV_64F);
    newCameraMatrix.at<double>(0,0) = focal;
    newCameraMatrix.at<double>(0,2) = centerx;
    newCameraMatrix.at<double>(1,1) = focal;
    newCameraMatrix.at<double>(1,2) = centery;

    cv::Mat map1, map2;

    cv::VideoCapture cap(FLAGS_video);
    int video_index = 0;
    cv::Mat image_in, image_distorted;
    while ( cap.read(image_in) )
    {
        if ( image_in.channels() == 3 ) cv::cvtColor( image_in, image_distorted, cv::COLOR_BGR2GRAY );
        else image_in.copyTo(image_distorted);
        
        if ( FLAGS_rotate == 1 ) cv::rotate(image_distorted, image_distorted, cv::ROTATE_90_CLOCKWISE);
        else if ( FLAGS_rotate == 2 ) cv::rotate(image_distorted, image_distorted, cv::ROTATE_180);
        else if ( FLAGS_rotate == 3 ) cv::rotate(image_distorted, image_distorted, cv::ROTATE_90_COUNTERCLOCKWISE);
        
        if ( video_index == 0 )
        {
            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::noArray(), newCameraMatrix, image_distorted.size(), CV_32FC1, map1, map2 );
        }
        
        cv::Mat image_undistorted;
        cv::remap( image_distorted, image_undistorted, map1, map2, cv::INTER_CUBIC );
        
        char path[1024];
        sprintf(path,"%s/%06d.png",FLAGS_output.c_str(),video_index);
        cv::imwrite( path, image_undistorted );
        
        video_index++;
    }
    
    FILE *intrinsicsout = fopen( FLAGS_intrinsicsout.c_str(), "w" );
    fprintf( intrinsicsout, "%.15lf %.15lf %.15lf", focal, centerx, centery );
    fclose( intrinsicsout );
}

