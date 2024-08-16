
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <gflags/gflags.h>

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"
#include <sphericalsfm/so3.h>

using namespace sphericalsfm;
using namespace sphericalsfmtools;

DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery)");
DEFINE_string(images, "", "Path to video or image search pattern like frame%06d.png");
DEFINE_string(output, "", "Path to output directory");
DEFINE_double(inlierthresh, 2.0, "Inlier threshold in pixels");
DEFINE_double(minrot, 1.0, "Minimum rotation between keyframes");
DEFINE_int32(mininliers, 100, "Minimum number of inliers to accept a loop closure");
DEFINE_int32(numbegin, 30, "Number of frames at beginning of sequence to use for loop closure");
DEFINE_int32(numend, 30, "Number of frames at end of sequence to use for loop closure");
DEFINE_bool(bestonly, false, "Accept only the best loop closure");
DEFINE_bool(inward, false, "Cameras are inward facing");
DEFINE_bool(sequential, false, "Images are sequential");

int main( int argc, char **argv )
{
    srand(0);

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    double focal, centerx, centery;
    std::ifstream intrinsicsf( FLAGS_intrinsics );
    intrinsicsf >> focal >> centerx >> centery;
    
    std::cout << "intrinsics : " << focal << ", " << centerx << ", " << centery << "\n";

    Intrinsics intrinsics(focal,centerx,centery);
    
    std::vector<Keyframe> keyframes;
    std::vector<ImageMatch> image_matches;
    if ( true ) //!read_feature_tracks( FLAGS_output, keyframes, image_matches ) )
    {
        std::cout << "tracking features in images: " << FLAGS_images << "\n";

        detect_features( FLAGS_images, keyframes );

        std::cout << "detecting loop closures\n";
        std::vector<ImageMatch> all_image_matches;
        match_exhaustive( keyframes, all_image_matches );
        int loop_closure_count = estimate_pairwise( intrinsics, keyframes, all_image_matches,
                            FLAGS_inlierthresh, FLAGS_mininliers, FLAGS_inward, image_matches );
        if ( loop_closure_count == 0 ) 
        {
            std::cout << "error: no loop closures found\n";
            exit(1);
        }

        write_feature_tracks( FLAGS_output, keyframes, image_matches ); 
    }
    else
    {
        read_images( FLAGS_images, keyframes );
    }

    std::cout << "initializing rotations\n";
    std::vector<Eigen::Matrix3d> rotations;
    if ( FLAGS_sequential ) {
        initialize_rotations_sequential( keyframes.size(), image_matches, rotations );
    } else {
        image_matches = filter_image_matches( image_matches, 2.*M_PI/180. );
        initialize_rotations_gopt( keyframes.size(), image_matches, rotations );
    }

    SfM pre_loop_closure_sfm( intrinsics );
    build_sfm( keyframes, image_matches, rotations, pre_loop_closure_sfm, true, true, FLAGS_inward );
    pre_loop_closure_sfm.WriteCameraCentersOBJ( FLAGS_output + "/pre-loop-cameras.obj" );
    exit(0);

    std::cout << "refining rotations\n";
    refine_rotations( keyframes.size(), image_matches, rotations );

    std::cout << "building sfm\n";
    SfM sfm( intrinsics );
    build_sfm( keyframes, image_matches, rotations, sfm, true, true, FLAGS_inward );
    
    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-spherical-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-spherical-ba.obj" );
    
    sfm.Optimize();
    sfm.Retriangulate();
    sfm.Optimize();

    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-ba.obj" );

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

    std::vector<int> keyframe_indices(keyframes.size());
    for ( int i = 0; i < keyframes.size(); i++ ) keyframe_indices[i] = keyframes[i].index;
    sfm.WritePoses( FLAGS_output + "/poses.txt", keyframe_indices );
    sfm.WritePointsOBJ( FLAGS_output + "/points.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras.obj" );

    sfm.WriteCOLMAP( FLAGS_output, keyframes[0].image.cols, keyframes[0].image.rows );
    
    //show_reprojection_error( keyframes, sfm );
}

