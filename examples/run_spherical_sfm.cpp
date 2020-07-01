
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <gflags/gflags.h>

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"

using namespace sphericalsfm;
using namespace sphericalsfmtools;

DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery)");
DEFINE_string(video, "", "Path to video or image search pattern like frame%06d.png");
DEFINE_string(output, "", "Path to output directory");
DEFINE_double(inlierthresh, 2.0, "Inlier threshold in pixels");
DEFINE_double(minrot, 1.0, "Minimum rotation between keyframes");
DEFINE_int32(mininliers, 100, "Minimum number of inliers to accept a loop closure");
DEFINE_int32(numbegin, 30, "Number of frames at beginning of sequence to use for loop closure");
DEFINE_int32(numend, 30, "Number of frames at end of sequence to use for loop closure");
DEFINE_bool(bestonly, false, "Accept only the best loop closure");
DEFINE_bool(noloopclosure, false, "Allow there to be no loop closures");

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
    if ( !read_feature_tracks( FLAGS_output, keyframes, image_matches ) )
    {
        std::cout << "tracking features in video: " << FLAGS_video << "\n";

        build_feature_tracks( intrinsics, FLAGS_video, keyframes, image_matches, FLAGS_inlierthresh, FLAGS_minrot );
        write_feature_tracks( FLAGS_output, keyframes, image_matches ); 
    }
    else
    {
        read_images( FLAGS_video, keyframes );
    }
    
    std::cout << "detecting loop closures\n";
    int loop_closure_count = make_loop_closures( intrinsics, keyframes, image_matches, FLAGS_inlierthresh, FLAGS_mininliers, FLAGS_numbegin, FLAGS_numend, FLAGS_bestonly );
    if ( FLAGS_bestonly ) std::cout << "only using best loop closure\n";
    if ( !FLAGS_noloopclosure && loop_closure_count == 0 ) 
    {
        std::cout << "error: no loop closures found\n";
        exit(1);
    }
    
    std::cout << "initializing rotations\n";
    std::vector<Eigen::Matrix3d> rotations;
    initialize_rotations( keyframes.size(), image_matches, rotations );

    SfM pre_loop_closure_sfm( intrinsics );
    build_sfm( keyframes, image_matches, rotations, pre_loop_closure_sfm );
    pre_loop_closure_sfm.WriteCameraCentersOBJ( FLAGS_output + "/pre-loop-cameras.obj" );

    std::cout << "refining rotations\n";
    refine_rotations( keyframes.size(), image_matches, rotations );

    std::cout << "building sfm\n";
    SfM sfm( intrinsics );
    build_sfm( keyframes, image_matches, rotations, sfm );
    
    sfm.WritePointsOBJ( FLAGS_output + "/pre-ba-points.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/pre-ba-cameras.obj" );
    
    std::cout << "running optimization\n";
    sfm.Optimize();
    std::cout << "done.\n";
    
    std::vector<int> keyframe_indices(keyframes.size());
    for ( int i = 0; i < keyframes.size(); i++ ) keyframe_indices[i] = keyframes[i].index;
    sfm.WritePoses( FLAGS_output + "/poses.txt", keyframe_indices );
    sfm.WritePointsOBJ( FLAGS_output + "/points.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras.obj" );
}

