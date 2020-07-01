
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include <gflags/gflags.h>

#include "stereo_panorama_tools.h"

using namespace sphericalsfm;
using namespace stereopanotools;

DEFINE_string(intrinsics, "", "Path to intrinsics (focal centerx centery)");
DEFINE_string(video, "", "Path to video or image search pattern like frame%06d.png");
DEFINE_string(output, "", "Path to output directory");
DEFINE_int32(width, 8192, "Width of output panorama");
DEFINE_bool(loop, true, "Trajectory is a closed loop");

int main( int argc, char **argv )
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    double focal, centerx, centery;
    std::ifstream intrinsicsf( FLAGS_intrinsics );
    intrinsicsf >> focal >> centerx >> centery;
    
    std::cout << "intrinsics : " << focal << ", " << centerx << ", " << centery << "\n";

    Intrinsics intrinsics(focal,centerx,centery);
    make_stereo_panoramas( intrinsics, FLAGS_video, FLAGS_output, FLAGS_width, FLAGS_loop );
}
