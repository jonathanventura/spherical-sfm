#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <sphericalsfm/sfm.h>

namespace sphericalsfmtools {
    typedef std::pair<size_t,size_t> Match;
    typedef std::map<size_t,size_t> Matches;

    cv::Vec3b sample_image(const cv::Mat& img, cv::Point2f pt);

    struct Features
    {
        std::vector<int> tracks;
        std::vector<cv::Point2f> points;
        std::vector<cv::Vec3b> colors;
        cv::Mat descs;
        Features() : descs(0,128,CV_32F) { }
        int size() const { return points.size(); }
        bool empty() const { return points.empty(); }
    };

    struct Keyframe
    {
        int index;
        std::string name;
        Features features;
        cv::Mat color_image;
        cv::Mat image;
        Keyframe( const int _index, const std::string &_name, const Features &_features ) :
        index(_index), name(_name), features(_features) { }
    };

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

    class DetectorTracker
    {
    protected:
        double min_dist; // minimum distance between existing points and detecting points
        const int xradius; // horizontal tracking radius
        const int yradius; // vertical tracking radius
        cv::Ptr<cv::SIFT> sift;
    public:
        DetectorTracker( double _min_dist=0, double _xradius=0, double _yradius=0 );
        
        void detect( const cv::Mat &gray_image, const cv::Mat &image, Features &features );
        void track( cv::Mat &image0, cv::Mat &image1,
                    const Features &features0, Features &features1,
                    Matches &m01 );
    };

    void initialize_rotations( int num_cameras, const std::vector<ImageMatch> &image_matches, bool sequential, std::vector<Eigen::Matrix3d> &rotations );

    void match( const Features &features0, const Features &features1, Matches &m01, double ratio = 0.75 );
    void detect_features( const std::string &videopath, std::vector<Keyframe> &keyframes );
    int estimate_pairwise( const sphericalsfm::Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const bool inward, std::vector<ImageMatch> &image_matches_out );
    int estimate_pairwise_five_point( const sphericalsfm::Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches,
                        const double inlier_threshold, const int min_num_inliers, std::vector<ImageMatch> &image_matches_out );
    void match_exhaustive( const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches );
    void build_feature_tracks( const sphericalsfm::Intrinsics &intrinsics, const std::string &videopath,
                              std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold, const double min_rot,
                              const bool inward = false );

    void find_largest_connected_component( std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches );

    int make_loop_closures( const sphericalsfm::Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end, const bool best_only,
                            const bool inward = false );

    std::vector<ImageMatch> filter_image_matches( std::vector<ImageMatch> &image_matches, double err_thresh_rad );
    void initialize_rotations_sequential( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations );
    void initialize_rotations_gopt( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations );
    double refine_rotations( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations );

    void build_sfm( std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
                   sphericalsfm::SfM &sfm, bool spherical = true, bool merge = true, bool inward = false, int fix_camera = 0 );

    void show_reprojection_error( std::vector<Keyframe> &keyframes, sphericalsfm::SfM &sfm );

    bool find_best_focal_length_opt( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal );
    bool find_best_focal_length_grid( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 int num_steps,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal );
    bool find_best_focal_length_random( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 int num_trials,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal );
}
