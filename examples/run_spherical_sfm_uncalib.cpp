
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
#include "colmap.h"
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
DEFINE_bool(colmap, false, "Load feature matches from COLMAP");

int main( int argc, char **argv )
{
    srand(0);
    
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int width, height;
    double focal_guess;
    double centerx, centery;
    std::vector<Keyframe> keyframes;
    std::vector<ImageMatch> all_image_matches, image_matches;
    if ( FLAGS_colmap )
    {
        COLMAP::Database db( FLAGS_output + "/database.db" );
        db.read();
        int index = 0;
        for ( auto it : db.images )
        {
            std::cout << it.second.name << "\n";
            cv::Mat image = cv::imread(FLAGS_images + "/" + it.second.name);
            if ( index == 0 )
            {
                width = image.cols;
                height = image.rows;
            } else {
                assert( ( width == image.cols ) && ( height == image.rows ) );
            }
            Features features;
            for ( auto kp : it.second.keypoints )
            {
                cv::Point2f pt( kp.x(0), kp.x(1) );
                features.points.push_back( pt );
                features.descs.push_back( cv::Mat::zeros(1,128,CV_32F) );
                features.colors.push_back( sample_image( image, pt ) );
            }
            Keyframe kf( index++, it.second.name, features );
            keyframes.push_back( kf );
        }
        for ( auto m : db.matches )
        {
            int index0 = m.image_id1-1;
            int index1 = m.image_id2-1;
            Matches matches;
            for ( auto mm : m.matches )
            {
                matches[mm.first] = mm.second;
            }
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            ImageMatch im( index0, index1, matches, R );
            all_image_matches.push_back(im);
        }
    } else {
        cv::VideoCapture cap(FLAGS_images);
        cv::Mat image0_in;
        if ( !cap.read(image0_in) )
        {
            std::cout << "error: could not read single frame from " << FLAGS_images << "\n";
            exit(1);
        }
        width = image0_in.cols;
        height = image0_in.rows;

        std::cout << "extract features from images: " << FLAGS_images << "\n";
        detect_features( FLAGS_images, keyframes );
        std::cout << "matching images\n";
        match_exhaustive( keyframes, all_image_matches );
    }

    focal_guess = (width+height)/2;
    centerx = width/2;
    centery = height/2;
    std::cout << "initial focal: " << focal_guess << "\n";
    Intrinsics intrinsics_guess(focal_guess,centerx,centery);
    int loop_closure_count = estimate_pairwise( intrinsics_guess, keyframes, all_image_matches,
                        FLAGS_inlierthresh, FLAGS_mininliers, FLAGS_inward, image_matches );
    if ( loop_closure_count == 0 ) 
    {
        std::cout << "error: no matches found\n";
        exit(1);
    }

    std::cout << "had " << keyframes.size() << " keyframes and " << image_matches.size() << " image matches\n";
    find_largest_connected_component( keyframes, image_matches );
    std::cout << "now have " << keyframes.size() << " keyframes and " << image_matches.size() << " image matches\n";
    
    for ( auto im : image_matches )
    {
        std::cout << im.index0 << " " << im.index1 << " " << so3ln(im.R).transpose() << "\n";
    }
    const double min_focal = focal_guess/4;
    const double max_focal = focal_guess*2;
    const int num_steps = 64;
    const int num_trials = 1024;
    const double min_total_rot = 270*M_PI/180.;
    const double max_total_rot = 450*M_PI/180.;
    std::vector<Eigen::Matrix3d> rotations;
    double focal_new;
    //bool success = find_best_focal_length_opt( keyframes.size(), 
            //image_matches, FLAGS_inward, FLAGS_sequential, focal_guess,
            //min_focal, max_focal,
            //rotations,
            //focal_new );
    //bool success = find_best_focal_length_grid( keyframes.size(), 
            //image_matches, FLAGS_inward, FLAGS_sequential, focal_guess,
            //min_focal, max_focal, num_steps,
            //rotations,
            //focal_new );
    bool success = find_best_focal_length_random( keyframes.size(), 
            image_matches, FLAGS_inward, FLAGS_sequential, focal_guess,
            min_focal, max_focal, num_trials,
            rotations,
            focal_new );
    if ( !success )
    {
        std::cout << "ERROR: could not find any acceptable focal length\n";
        exit(1);
    }
    std::cout << " best focal: " << focal_new << "\n";
    Intrinsics intrinsics(focal_new,centerx,centery);
    //exit(0);

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

    sfm.WriteCOLMAP( FLAGS_output, width, height );
        

    std::string calib_path = FLAGS_output + "/calib.txt";
    FILE *calibf = fopen(calib_path.c_str(), "w");
    fprintf(calibf, "%0.15f %0.15f %0.15f\n", sfm.GetFocal(), centerx, centery );
    fclose(calibf);
}

