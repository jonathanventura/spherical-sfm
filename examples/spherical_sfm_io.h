#pragma once

#include "spherical_sfm_tools.h"

namespace sphericalsfmtools {

    void write_feature_tracks( const std::string &outputpath, const std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches );
    bool read_feature_tracks( const std::string &outputpath, std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches );
    void read_images( const std::string &videopath, std::vector<Keyframe> &keyframes );
}
