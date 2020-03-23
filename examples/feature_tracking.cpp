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

#include "feature_tracking.h"

namespace feature_tracking {
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(2000);

    class DetectorTracker
    {
    protected:
        int next_feature_index;
        double min_dist; // minimum distance between existing points and detecting points
        const int xradius; // horizontal tracking radius
        const int yradius; // vertical tracking radius
    public:
        DetectorTracker( double _min_dist, double _xradius, double _yradius ) :
        next_feature_index(0), min_dist(_min_dist), xradius(_xradius), yradius(_yradius) { }
        
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
                    features.push_back(Feature( next_feature_index++, keypoints[i].pt.x,keypoints[i].pt.y,descs.row(i)));
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
                    
                    features.push_back( Feature( next_feature_index++, keypoints[i].pt.x, keypoints[i].pt.y, descs.row(i) ) );
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
                
                features1.push_back(Feature( features0[i].index, nextPts[i].x, nextPts[i].y, features0[i].descriptor ) );
                m01[i] = features1.size()-1;
            }
        }

    };


    void build_feature_tracks( const sphericalsfm::Intrinsics &intrinsics, std::string &videopath,
                              std::vector<Features> &features, std::vector<RelativePose> &relative_poses,
                              const double min_rot )
    {
        bool inward = false;
        
        std::cout << "K:\n" << intrinsics.getK() << "\n";
        std::cout << "Kinv:\n" << intrinsics.getKinv() << "\n";
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
        
        sphericalsfm::PreemptiveRANSAC<sphericalsfm::RayPairList,sphericalsfm::Estimator> ransac( 100 );
        ransac.inlier_threshold = 2*Kinv(0);

        std::vector<sphericalsfm::Estimator*> estimators( 2000 );
        for ( size_t i = 0; i < estimators.size(); i++ )
        {
            estimators[i] = new sphericalsfm::SphericalEstimator;
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
            sphericalsfm::RayPairList ray_pair_list;
            ray_pair_list.reserve( m01.size() );
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                size_t index0 = it->first;
                size_t index1 = it->second;

//                std::cout << features0[index0].x << features0[index0].y << "\t" << features1[index1].x << features1[index1].y << "\n";
                Eigen::Vector3d loc0 = Eigen::Vector3d( features0[index0].x, features0[index0].y, 1 );
                Eigen::Vector3d loc1 = Eigen::Vector3d( features1[index1].x, features1[index1].y, 1 );
                loc0 = Kinv * loc0;
                loc1 = Kinv * loc1;
                
                sphericalsfm::Ray u;
                u.head(3) = loc0;
                sphericalsfm::Ray v;
                v.head(3) = loc1;

                ray_pair_list.push_back( std::make_pair( u, v ) );
            }
            
            sphericalsfm::SphericalEstimator *best_estimator = NULL;
            
            std::cout << "running ransac on " << ray_pair_list.size() << " matches\n";
            
            ninliers = ransac.compute( ray_pair_list.begin(), ray_pair_list.end(), estimators, (sphericalsfm::Estimator**)&best_estimator, inliers );
            
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
            
//            writeMatches( datapath.c_str(), last_index, index, new_m01 );
//            writeRelativeRotation( datapath.c_str(), last_index, index, best_estimator_r );
            relative_poses.push_back( RelativePose( index-1, index, sphericalsfm::so3exp(best_estimator_r) ) );
            index++;
        }
    }
}
