
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"
#include <sphericalsfm/so3.h>

using namespace sphericalsfm;
using namespace sphericalsfmtools;

DEFINE_string(images, "", "Path to video or image search pattern like frame%06d.png");
DEFINE_string(output, "", "Path to output directory");
DEFINE_double(inlierthresh, 2.0, "Inlier threshold in pixels");
DEFINE_int32(mininliers, 100, "Minimum number of inliers to accept a loop closure");
DEFINE_int32(numbegin, 30, "Number of frames at beginning of sequence to use for loop closure");
DEFINE_int32(numend, 30, "Number of frames at end of sequence to use for loop closure");
DEFINE_bool(inward, false, "Cameras are inward facing");
DEFINE_bool(sequential, false, "Images are sequential");
DEFINE_bool(generalba, false, "Run general bundle adjustment step");

int main( int argc, char **argv )
{
    srand(0);

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    cv::VideoCapture cap(FLAGS_images);
    cv::Mat image0_in;
    if ( !cap.read(image0_in) )
    {
        std::cout << "error: could not read single frame from " << FLAGS_images << "\n";
        exit(1);
    }
    int width = image0_in.cols;
    int height = image0_in.rows;
    
    double focal_guess = (width+height)/2;//sqrt(width*width+height*height);
    double centerx = width/2;
    double centery = height/2;
    std::cout << "initial focal: " << focal_guess << "\n";
    Intrinsics intrinsics_guess(focal_guess,centerx,centery);
    int min_rot_pixels = 35;
    // theta = 2*atan2(w,2*f)
    double min_rot = 0;//2*atan2(min_rot_pixels,2*focal_guess)*180/M_PI;
    //std::cout << "min_rot: " << min_rot << " [deg]\n";
    //std::cout << "min_rot: " << min_rot*M_PI/180. << " [rad]\n";
    
    std::vector<Keyframe> keyframes;
    std::vector<ImageMatch> image_matches;
    if ( !read_feature_tracks( FLAGS_output, keyframes, image_matches ) )
    {
        std::cout << "extract features from images: " << FLAGS_images << "\n";

        detect_features( FLAGS_images, keyframes );
        std::cout << "matching images\n";
        int loop_closure_count = match_exhaustive( intrinsics_guess, keyframes, image_matches,
                            FLAGS_inlierthresh, FLAGS_mininliers, FLAGS_inward );
        if ( loop_closure_count == 0 ) 
        {
            std::cout << "error: no matches found\n";
            exit(1);
        }

        write_feature_tracks( FLAGS_output, keyframes, image_matches ); 
    }
    else
    {
        read_images( FLAGS_images, keyframes );
    }
    
    const double min_focal = focal_guess/4;
    const double max_focal = focal_guess*2;
    const int num_steps = 64;
    const double min_total_rot = 270*M_PI/180.;
    const double max_total_rot = 450*M_PI/180.;
    std::vector<Eigen::Matrix3d> rotations;
    double focal_new;
    bool success = find_best_focal_length_opt( keyframes.size(), 
            image_matches, FLAGS_inward, FLAGS_sequential, focal_guess,
            min_focal, max_focal,
            rotations,
            focal_new );
    //bool success = find_best_focal_length_grid( keyframes.size(), 
            //image_matches, FLAGS_inward, FLAGS_sequential, focal_guess,
            //min_focal, max_focal, num_steps,
            //rotations,
            //focal_new );
    if ( !success )
    {
        std::cout << "ERROR: could not find any acceptable focal length\n";
        exit(1);
    }
    std::cout << " best focal: " << focal_new << "\n";
    //exit(0);
    Intrinsics intrinsics(focal_new,centerx,centery);

    std::cout << "building sfm\n";
    SfM sfm( intrinsics );
    build_sfm( keyframes, image_matches, rotations, sfm, true, true, FLAGS_inward );
    sfm.SetFocalFixed(false);
    
    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-spherical-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-spherical-ba.obj" );
    
    sfm.Optimize();
    sfm.Retriangulate();
    sfm.Optimize();

    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-ba.obj" );
    
    std::cout << "focal after spherical BA: " << sfm.GetFocal() << "\n";
    

    if ( FLAGS_generalba )
    {
        // --- general SFM ---
        // unfix translations
        for ( int i = 1; i < sfm.GetNumCameras(); i++ )
        {
            sfm.SetTranslationFixed(i,false);
        }

        std::cout << "running general optimization\n";
        sfm.Optimize();
        sfm.Normalize( FLAGS_inward );
        sfm.Retriangulate();
        sfm.Optimize();
        sfm.Normalize( FLAGS_inward );
        std::cout << "done.\n";
        std::cout << "focal after general BA: " << sfm.GetFocal() << "\n";
    }

    std::vector<int> keyframe_indices(keyframes.size());
    for ( int i = 0; i < keyframes.size(); i++ ) keyframe_indices[i] = keyframes[i].index;
    sfm.WritePoses( FLAGS_output + "/poses.txt", keyframe_indices );
    sfm.WritePointsOBJ( FLAGS_output + "/points.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras.obj" );

    sfm.WriteCOLMAP( FLAGS_output, keyframes[0].image.cols, keyframes[0].image.rows );
        

    std::string calib_path = FLAGS_output + "/calib.txt";
    FILE *calibf = fopen(calib_path.c_str(), "w");
    fprintf(calibf, "%0.15f %0.15f %0.15f\n", sfm.GetFocal(), centerx, centery );
    fclose(calibf);
}

