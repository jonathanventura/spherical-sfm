#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include <sphericalsfm/sfm.h>

namespace feature_tracking {
    typedef std::pair<size_t,size_t> Match;
    typedef std::map<size_t,size_t> Matches;

    struct Feature
    {
        int index;
        float x, y;
        cv::Mat descriptor;
        Feature( int _index, float _x, float _y, cv::Mat _descriptor ) : index(_index), x(_x), y(_y) {
            _descriptor.copyTo(descriptor);
        }
        
    };
    typedef std::vector<Feature> Features;

    struct RelativePose
    {
        int index0, index1;
        Eigen::Matrix3d R;
        RelativePose( const int _index0, const int _index1, const Eigen::Matrix3d &_R ) :
        index0(_index0), index1(_index1), R(_R) { }
    };

    void build_feature_tracks( const sphericalsfm::Intrinsics &intrinsics, std::string &videopath,
                              std::vector<Features> &features, std::vector<RelativePose> &relative_poses,
                              const double min_rot = 1.0 );
}
