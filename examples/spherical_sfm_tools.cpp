#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video.hpp>

#include <cstdio>
#include <iostream>
#include <sys/stat.h>

#include <cassert>

#include <sphericalsfm/spherical_estimator.h>
#include <sphericalsfm/preemptive_ransac.h>
#include <sphericalsfm/so3.h>

#include <openMVG/multiview/rotation_averaging_l1.hpp>
#include <openMVG/multiview/rotation_averaging_l2.hpp>

#include "spherical_sfm_tools.h"

using namespace sphericalsfm;

namespace sphericalsfmtools {
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(2000);

    class DetectorTracker
    {
    protected:
        double min_dist; // minimum distance between existing points and detecting points
        const int xradius; // horizontal tracking radius
        const int yradius; // vertical tracking radius
    public:
        DetectorTracker( double _min_dist=0, double _xradius=0, double _yradius=0 ) :
        min_dist(_min_dist), xradius(_xradius), yradius(_yradius) { }
        
        void detect( cv::Mat &image, Features &features )
        {
            assert( image.type() == CV_8UC1 );
            
            std::vector<cv::KeyPoint> keypoints;
            sift->detect( image, keypoints );
            std::cout << keypoints.size() << "\n";
            
            if ( keypoints.empty() ) return;
            
            if ( features.empty() )
            {
                cv::Mat descs;
                sift->compute( image, keypoints, descs );
                
                for  ( int i = 0; i < keypoints.size(); i++ )
                {
                    features.push_back( Feature( keypoints[i].pt.x, keypoints[i].pt.y, descs.row(i) ) );
                }
            }
            else
            {
                cv::Mat descs;
                sift->compute( image, keypoints, descs );

                std::vector<bool> should_add( keypoints.size() );
                for ( int i = 0; i < keypoints.size(); i++ ) should_add[i] = true;
            
                cv::Mat pts(features.size(),2,CV_32F);
                for ( int i = 0; i < features.size(); i++ )
                {
                    pts.at<float>(i,0) = features[i].x;
                    pts.at<float>(i,1) = features[i].y;
                }
                cv::flann::GenericIndex< cvflann::L2_Simple<float> > index( pts, cvflann::KDTreeIndexParams( 1 ) );
                
                for ( int i = 0; i < keypoints.size(); i++ )
                {
                    cv::Mat query(1,2,CV_32F);
                    query.at<float>(0,0) = keypoints[i].pt.x;
                    query.at<float>(0,1) = keypoints[i].pt.y;
                    cv::Mat indices(1,1,CV_32S);
                    cv::Mat dists(1,1,CV_32F);
                    int n = index.radiusSearch( query, indices, dists, min_dist, cvflann::SearchParams() );
                    should_add[i] = ( n == 0 );
                }

                for ( int i = 0; i < keypoints.size(); i++ )
                {
                    if ( !should_add[i] ) continue;
                    
                    features.push_back( Feature( keypoints[i].pt.x, keypoints[i].pt.y, descs.row(i) ) );
                }
            }
        }

        void track( cv::Mat &image0, cv::Mat &image1,
                    const Features &features0, Features &features1,
                    Matches &m01 )
        {
            const size_t npts0 = features0.size();
            
            std::vector<cv::Point2f> prevPts(npts0);
            for ( int i = 0; i < npts0; i++ )
            {
                prevPts[i].x = features0[i].x;
                prevPts[i].y = features0[i].y;
            }
            std::vector<cv::Point2f> nextPts;
            std::vector<uchar> status;
            std::vector<float> err;
            
            cv::calcOpticalFlowPyrLK(image0, image1, prevPts, nextPts, status, err, cv::Size(xradius*2+1,yradius*2+1));
            
            for ( int i = 0; i < npts0; i++ )
            {
                if ( status[i] == 0 ) continue;
                
                features1.push_back( Feature( nextPts[i].x, nextPts[i].y, features0[i].descriptor ) );
                m01[i] = features1.size()-1;
            }
        }

    };

    void match( const Features &features0, const Features &features1, Matches &m01, double min_ratio = 0.75 )
    {
        static cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

        std::vector< std::vector<cv::DMatch> > matches;
        
        cv::Mat descs0(features0.size(),128,CV_32F);
        for ( int i = 0; i < features0.size(); i++ ) features0[i].descriptor.copyTo(descs0.row(i));
        cv::Mat descs1(features1.size(),128,CV_32F);
        for ( int i = 0; i < features1.size(); i++ ) features1[i].descriptor.copyTo(descs1.row(i));

        // p1 = query, p0 = train
        matcher->knnMatch( descs1, descs0, matches, 2);
        
        for ( int i = 0; i < matches.size(); i++ )
        {
            if ( matches[i][0].distance < min_ratio*matches[i][1].distance )
            {
                m01[matches[i][0].trainIdx] = matches[i][0].queryIdx;
            }
        }
    }

    void build_feature_tracks( const Intrinsics &intrinsics, const std::string &videopath,
                              std::vector<Features> &features, std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold, const double min_rot )
    {
        bool inward = false;
        
        Eigen::Matrix3d Kinv = intrinsics.getKinv();
        
        cv::VideoCapture cap(videopath);
        
        cv::Mat image0_in, image0;
        if ( !cap.read(image0_in) )
        {
            std::cout << "error: could not read single frame from " << videopath << "\n";
            exit(1);
        }
        if ( image0_in.channels() == 3 ) cv::cvtColor( image0_in, image0, cv::COLOR_BGR2GRAY );
        else image0 = image0_in;
        
        int xradius = image0.cols/48;
        int yradius = image0.rows/48;
        
        DetectorTracker detectortracker( 10*xradius, xradius, yradius );

        Features features0;
        detectortracker.detect( image0, features0 );
        features.push_back(features0);

        std::cout << "detected " << features0.size() << " features in image 0\n";
        
        PreemptiveRANSAC<RayPairList,Estimator> ransac( 100 );
        ransac.inlier_threshold = inlier_threshold/intrinsics.focal;

        std::vector<Estimator*> estimators( 2000 );
        for ( size_t i = 0; i < estimators.size(); i++ )
        {
            estimators[i] = new SphericalEstimator;
        }

        cv::Mat image1_in, image1;
        int index = 1;
        while ( cap.read( image1_in ) )
        {
            if ( image1_in.channels() == 3 ) cv::cvtColor( image1_in, image1, cv::COLOR_BGR2GRAY );
            else image1 = image1_in;

            Features features1;
            
            Matches m01;
            
            detectortracker.track( image0, image1, features0, features1, m01 );
            std::cout << "tracked " << features1.size() << " features in image " << index << "\n";

            std::vector<bool> inliers;
            int ninliers;

            // find inlier matches using relative pose
            RayPairList ray_pair_list;
            ray_pair_list.reserve( m01.size() );
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                size_t index0 = it->first;
                size_t index1 = it->second;

                Eigen::Vector3d loc0 = Eigen::Vector3d( features0[index0].x, features0[index0].y, 1 );
                Eigen::Vector3d loc1 = Eigen::Vector3d( features1[index1].x, features1[index1].y, 1 );
                loc0 = Kinv * loc0;
                loc1 = Kinv * loc1;
                
                Ray u;
                u.head(3) = loc0;
                Ray v;
                v.head(3) = loc1;

                ray_pair_list.push_back( std::make_pair( u, v ) );
            }
            
            SphericalEstimator *best_estimator = NULL;
            
            std::cout << "running ransac on " << ray_pair_list.size() << " matches\n";
            
            ninliers = ransac.compute( ray_pair_list.begin(), ray_pair_list.end(), estimators, (Estimator**)&best_estimator, inliers );
            
            fprintf( stdout, "%d: %lu matches and %d inliers (%0.2f%%)\n", index, m01.size(), ninliers, (double)ninliers/(double)m01.size()*100 );

            Eigen::Vector3d best_estimator_t;
            Eigen::Vector3d best_estimator_r;
            best_estimator->decomposeE( inward, ray_pair_list.begin(), ray_pair_list.end(), inliers, best_estimator_r, best_estimator_t );
            std::cout << best_estimator_r.transpose() << "\n";

            // check for minimum rotation
            if ( best_estimator_r.norm() < min_rot * M_PI / 180. ) continue;

            Matches m01inliers;
            size_t i = 0;
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                size_t index0 = it->first;
                size_t index1 = it->second;
                if ( inliers[i++] ) m01inliers[index0] = index1;
            }
            
            image1.copyTo(image0);
            
            // only propagate inlier points
            Matches new_m01;
            features0 = Features();
            for ( Matches::const_iterator it = m01inliers.begin(); it != m01inliers.end(); it++ )
            {
                new_m01[it->first] = features0.size();
                size_t index1 = it->second;
                features0.push_back(features1[index1]);
            }
            
            std::cout << "features0 now has " << features0.size() << " features\n";
            // also detect new points that are some distance away from matched points
            detectortracker.detect( image0, features0 );
            std::cout << "after sift, features0 now has " << features0.size() << " features\n";

            features.push_back(features0);
            
            image_matches.push_back( ImageMatch( index-1, index, new_m01, so3exp(best_estimator_r) ) );
            index++;
        }
    }

    void make_loop_closures( const Intrinsics &intrinsics, const std::vector<Features> &features, std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end )
    {
        Eigen::Matrix3d K0inv = intrinsics.getKinv();
        
        PreemptiveRANSAC<RayPairList,Estimator> ransac;
        ransac.inlier_threshold = inlier_threshold*K0inv(0);

        std::vector<Estimator*> estimators( 2000 );
        for ( size_t i = 0; i < estimators.size(); i++ )
        {
            estimators[i] = new SphericalEstimator;
        }
                
        int count0 = 0;
        for ( int index0 = 0; index0 < features.size(); index0++ )
        {
            if ( ++count0 > num_frames_begin ) break;

            const Features &features0 = features[index0];
            
            int count1 = 0;
            for ( int index1 = features.size()-1; index1 >= index0+1; index1-- )
            {
                if ( ++count1 > num_frames_end ) break;
                            
                const Features &features1 = features[index1];

                Matches m01;
                
                match( features0, features1, m01 );

                std::vector<bool> inliers;
                int ninliers;

                // find inlier matches using relative pose
                RayPairList ray_pair_list;
                ray_pair_list.reserve( m01.size() );
                for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
                {
                    size_t index0 = it->first;
                    size_t index1 = it->second;

                    Eigen::Vector3d loc0 = Eigen::Vector3d( features0[index0].x, features0[index0].y, 1 );
                    Eigen::Vector3d loc1 = Eigen::Vector3d( features1[index1].y, features1[index1].y, 1 );
                    
                    loc0 = K0inv * loc0;
                    loc1 = K0inv * loc1;
                    
                    Ray u;
                    u.head(3) = loc0;
                    Ray v;
                    v.head(3) = loc1;

                    ray_pair_list.push_back( std::make_pair( u, v ) );
                }
                
                SphericalEstimator *best_estimator = NULL;
                
                if ( m01.empty() ) ninliers = 0;
                else ninliers = ransac.compute( ray_pair_list.begin(), ray_pair_list.end(), estimators, (Estimator**)&best_estimator, inliers );
                fprintf( stdout, "%d %d: %lu matches and %d inliers (%0.2f%%)\n",
                        index0, index1, m01.size(), ninliers, (double)ninliers/(double)m01.size()*100 );

                if ( ninliers > min_num_inliers )
                {
                    Eigen::Vector3d best_estimator_t;
                    Eigen::Vector3d best_estimator_r;
                    best_estimator->decomposeE( false, ray_pair_list.begin(), ray_pair_list.end(), inliers, best_estimator_r, best_estimator_t );
                    
                    std::cout << best_estimator_r.transpose() << "\n";

                    Matches m01inliers;
                    size_t i = 0;
                    for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
                    {
                        size_t index0 = it->first;
                        size_t index1 = it->second;
                        if ( inliers[i++] ) m01inliers[index0] = index1;
                    }
                    
                    image_matches.push_back(ImageMatch(index0, index1, m01inliers, so3exp(best_estimator_r)));
                }
            }
        }
    }

    void initialize_rotations( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations )
    {
        openMVG::rotation_averaging::RelativeRotations RelRs;
        
        rotations.resize(num_cameras);
        for ( int i = 0 ; i < num_cameras; i++ ) rotations[i] = Eigen::Matrix3d::Identity();
        
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        rotations[0] = R;
        for ( int index = 1; index < num_cameras; index++ )
        {
            for ( int i = 0; i < image_matches.size(); i++ )
            {
                if ( image_matches[i].index0 == index-1 && image_matches[i].index1 == index )
                {
                    R = image_matches[i].R * R;
                    rotations[index] = R;
                    break;
                }
            }
        }
        
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            std::cout << image_matches[i].index0 << "\t" << image_matches[i].index1 << "\t" << image_matches[i].R << "\n";
            RelRs.push_back( openMVG::rotation_averaging::RelativeRotation( image_matches[i].index0, image_matches[i].index1, image_matches[i].R ) );
        }
        
//        openMVG::rotation_averaging::l2::L2RotationAveraging_Refine( RelRs, rotations );
    }

    void build_sfm( const std::vector<Features> &features, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
                   sphericalsfm::SfM &sfm )
    {
        std::vector< std::vector<int> > tracks(features.size());
        for ( int i = 0; i < features.size(); i++ )
        {
            tracks[i].resize( features[i].size() );
            for ( int j = 0; j < features[i].size(); j++ )
            {
                tracks[i][j] = -1;
            }
        }
        
        // add cameras
        for ( int index = 0; index < features.size(); index++ )
        {
            int camera = sfm.AddCamera( Pose( Eigen::Vector3d(0,0,-1), so3ln(rotations[index]) ) );
            sfm.SetRotationFixed( camera, (index==0) );
            sfm.SetTranslationFixed( camera, true );
        }

        for ( int i = 0; i < image_matches.size(); i++ )
        {
            const Matches &m01 = image_matches[i].matches;
            
            const int index0 = image_matches[i].index0;
            const int index1 = image_matches[i].index1;
            
            const Features &features0 = features[index0];
            const Features &features1 = features[index1];
            
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                const Feature &feature0 = features0[it->first];
                const Feature &feature1 = features1[it->second];
                
                int &track0 = tracks[index0][it->first];
                int &track1 = tracks[index1][it->second];
                
                Observation obs0( feature0.x-sfm.GetIntrinsics().centerx, feature0.y-sfm.GetIntrinsics().centery );
                Observation obs1( feature1.x-sfm.GetIntrinsics().centerx, feature1.y-sfm.GetIntrinsics().centery );

                if ( track0 != -1 && track1 == -1 )
                {
                    track1 = track0;
                    sfm.AddObservation( index1, track0, obs1 );
                }
                else if ( track0 == -1 && track1 != -1 )
                {
                    track0 = track1;
                    sfm.AddObservation( index0, track1, obs0 );
                }
                else if ( track0 == -1 && track1 == -1 )
                {
                    track0 = track1 = sfm.AddPoint( Eigen::Vector3d::Zero(), feature0.descriptor );

                    sfm.SetPointFixed( track0, false );
                    
                    sfm.AddObservation( index0, track0, obs0 );
                    sfm.AddObservation( index1, track1, obs1 );
                }
                else if ( track0 != -1 && track1 != -1 && ( track0 != track1 ) )
                {
                    // track1 will be removed
                    sfm.MergePoint( track0, track1 );
                    
                    // update all features with track1 and set to track0
                    for ( int i = 0; i < tracks.size(); i++ )
                    {
                        for ( int j = 0; j < tracks[j].size(); j++ )
                        {
                            if ( tracks[i][j] == track1 ) tracks[i][j] = track0;
                        }
                    }
                }
            }
        }
        
        sfm.Retriangulate();
    }

}
