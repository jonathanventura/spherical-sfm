
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include <gflags/gflags.h>

#include "feature_tracking.h"

DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery)");
DEFINE_string(video, "", "Path to video or image search pattern like frame%06d.png");
DEFINE_double(minrot, 1.0, "Minimum rotation between keyframes");

int main( int argc, char **argv )
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    double focal, centerx, centery;
    std::ifstream intrinsicsf( FLAGS_intrinsics );
    intrinsicsf >> focal >> centerx >> centery;
    
    std::cout << "intrinsics : " << focal << ", " << centerx << ", " << centery << "\n";

    sphericalsfm::Intrinsics intrinsics(focal,centerx,centery);
    
    std::cout << "tracking features in video: " << FLAGS_video << "\n";

    std::vector<feature_tracking::Features> features;
    std::vector<feature_tracking::RelativePose> relative_poses;
    feature_tracking::build_feature_tracks( intrinsics, FLAGS_video, features, relative_poses, FLAGS_minrot );
    
    std::cout << "detecting loop closures\n";
    
    
}
