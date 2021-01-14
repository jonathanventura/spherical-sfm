
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

#include <gflags/gflags.h>

#include "spherical_sfm_tools.h"
#include "spherical_sfm_io.h"
//#include <sphericalsfm/viewgraph.h>
#include <sphericalsfm/so3.h>

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
    build_sfm( keyframes, image_matches, rotations, sfm, true, true );
    
    std::cout << "adding view graph edges\n";
    for ( int i = 0; i < image_matches.size(); i++ )
    {
        //sfm.AddSphericalEdge( image_matches[i].index0, image_matches[i].index1, so3ln(image_matches[i].R) ); 
        sfm.AddEdge( image_matches[i].index0, image_matches[i].index1 );
    }

/*
    for ( int i = 0; i < keyframes.size()-1; i++ )
    {
        sfm.AddEdge( i, i+1 );
    }

    for ( int i = 0; i < FLAGS_numbegin; i++ )
    {
        for ( int j = keyframes.size()-1-FLAGS_numend; j < keyframes.size(); j++ )
        {
            sfm.AddEdge( i, j );
        }
    }
*/

/*
    for ( int i = 0; i < image_matches.size(); i++ )
    {
        Eigen::Matrix3d R = image_matches[i].R;
        Eigen::Vector3d t = R.col(2) - Eigen::Vector3d(0,0,1);
        Pose relpose( t, so3ln(R) );
        sfm.AddEdge( image_matches[i].index0, image_matches[i].index1, relpose ); 
        sfm.AddEdge( image_matches[i].index1, image_matches[i].index0, relpose.inverse() ); 
        if ( ( image_matches[i].index1 == image_matches[i].index0+1 ) && 
             ( image_matches[i].index1 != keyframes.size()-1 ) && 
             ( i != image_matches.size()-1 ) )
        {
            // add extra edge
            Eigen::Matrix3d R2 = image_matches[i+1].R;
            Eigen::Vector3d t2 = R2.col(2) - Eigen::Vector3d(0,0,1);
            Pose relpose2( t2, so3ln(R2) );

            Eigen::Matrix4d P = relpose2.P * relpose.P;
            Pose relpose3( P.block(0,3,3,1), so3ln(P.block(0,0,3,3)) );
            sfm.AddEdge( image_matches[i].index0, image_matches[i+1].index1, relpose3 );
            sfm.AddEdge( image_matches[i+1].index1, image_matches[i].index0, relpose3.inverse() );
        }
    }
*/
    
/*
    // add view triplets
    std::cout << "adding triplets\n";
    for ( int i = 0; i < sfm.GetNumCameras()-2; i++ )
    {
        int j = i+1;
        sfm.AddViewTriplet(i,j,j+1);
        std::cout << "\t" << i << " " << j << " " << j+1 << "\n";
        for ( int k = j+2; k < sfm.GetNumCameras(); k++ )
        {
            for ( int n = 0; n < image_matches.size(); n++ )
            {
                ImageMatch &m(image_matches[n]);
                if ( (m.index0 == i && m.index1 == k) ||
                     (m.index1 == i && m.index0 == k) ||
                     (m.index0 == j && m.index1 == k) || 
                     (m.index1 == j && m.index0 == k) )
                {
                    sfm.AddViewTriplet(i,j,k);
                    std::cout << "\t" << i << " " << j << " " << k << "\n";
                    break;
                }
            }
        }
    }
*/
    
    //sfm.RetriangulateRobust();
    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-spherical-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-spherical-ba.obj" );
    
    //std::cout << "running optimization\n";
    //sfm.OptimizeViewGraph();
    //sfm.OptimizeViewTriplets();
    //sfm.RetriangulateRobust();
    sfm.Optimize();
    sfm.Retriangulate();
    sfm.Optimize();
    //std::cout << "done.\n";

    sfm.WritePointsOBJ( FLAGS_output + "/points-pre-ba.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-ba.obj" );

    //std::cout << "detecting loop closures\n";
    //int loop_closure_count = make_3d_loop_closures( keyframes, sfm, FLAGS_mininliers, FLAGS_numbegin, FLAGS_numend );
    //exit(0);
    


    // unfix translations
    //sfm.SetRotationFixed(1,true);
    for ( int i = 1; i < sfm.GetNumCameras(); i++ )
    {
        sfm.SetTranslationFixed(i,false);
    }

    std::cout << "running general optimization\n";
    //sfm.Retriangulate();
    sfm.Optimize();
    sfm.Normalize();
    sfm.Retriangulate();
    sfm.Optimize();
    sfm.Normalize();
    std::cout << "done.\n";

    //sfm.WritePointsOBJ( FLAGS_output + "/points-pre-pose-graph.obj" );
    //sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras-pre-pose-graph.obj" );

    std::vector<int> keyframe_indices(keyframes.size());
    for ( int i = 0; i < keyframes.size(); i++ ) keyframe_indices[i] = keyframes[i].index;
    sfm.WritePoses( FLAGS_output + "/poses.txt", keyframe_indices );
    sfm.WritePointsOBJ( FLAGS_output + "/points.obj" );
    sfm.WriteCameraCentersOBJ( FLAGS_output + "/cameras.obj" );
        
    show_reprojection_error( keyframes, sfm );
}

