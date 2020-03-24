#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include <opencv2/core.hpp>

#include <sphericalsfm/sfm.h>

namespace sphericalsfmtools {
    typedef std::pair<size_t,size_t> Match;
    typedef std::map<size_t,size_t> Matches;

    struct Feature
    {
        float x, y;
        cv::Mat descriptor;
        Feature( float _x, float _y, cv::Mat _descriptor ) : x(_x), y(_y) {
            _descriptor.copyTo(descriptor);
        }
        
    };
    typedef std::vector<Feature> Features;

    struct ImageMatch
    {
        int index0, index1;
        Matches matches;
        Eigen::Matrix3d R;
        ImageMatch( const int _index0, const int _index1, const Matches &_matches, const Eigen::Matrix3d &_R ) :
        index0(_index0), index1(_index1), matches(_matches), R(_R) { }
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    void build_feature_tracks( const sphericalsfm::Intrinsics &intrinsics, const std::string &videopath,
                              std::vector<Features> &features, std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold, const double min_rot );

    void make_loop_closures( const sphericalsfm::Intrinsics &intrinsics, const std::vector<Features> &features, std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end );

    void initialize_rotations( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations );

    void build_sfm( const std::vector<Features> &features, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
                   sphericalsfm::SfM &sfm );
}
