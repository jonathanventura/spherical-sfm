#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video.hpp>

#include <cstdio>
#include <iostream>
#include <sys/stat.h>

#include <cassert>

#include <sphericalsfm/spherical_estimator.h>
#include <sphericalsfm/so3.h>
#include <sphericalsfm/rotation_averaging.h>
#include <sphericalsfm/uncalibrated_pose_graph.h>
#include <sphericalsfm/spherical_utils.h>

#include <RansacLib/ransac.h>

#include "spherical_sfm_tools.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>

#include <gopt/graph/view_graph.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


using namespace sphericalsfm;

namespace sphericalsfmtools {

    // from https://github.com/BAILOOL/ANMS-Codes/blob/master/C%2B%2B/QtProject/anms.h
    struct sort_pred {
        bool operator()(const std::pair<float,int> &left, const std::pair<float,int> &right) {
            return left.first > right.first;
        }
    };

    std::vector<cv::KeyPoint> brownANMS(std::vector<cv::KeyPoint> keyPoints, int numRetPoints) {
        std::vector<std::pair<float,int> > results;
        results.push_back(std::make_pair(FLT_MAX,0));
        for (unsigned int i = 1;i<keyPoints.size();++i){ //for every keyPoint we get the min distance to the previously visited keyPoints
            float minDist = FLT_MAX;
            for (unsigned int j=0;j<i;++j){
                float exp1 = (keyPoints[j].pt.x-keyPoints[i].pt.x);
                float exp2 = (keyPoints[j].pt.y-keyPoints[i].pt.y);
                float curDist = sqrt(exp1*exp1+exp2*exp2);
                minDist = std::min(curDist,minDist);
            }
            results.push_back(std::make_pair(minDist,i));
        }
        std::sort(results.begin(),results.end(),sort_pred()); //sorting by radius
        std::vector<cv::KeyPoint> kp;
        for (int i=0;i<numRetPoints;++i) kp.push_back(keyPoints[results[i].second]); //extracting numRetPoints keyPoints

        return kp;
    }

   void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                       const int numToKeep )
    {
      if( keypoints.size() < numToKeep ) { return; }
      std::sort( keypoints.begin(), keypoints.end(),
                 [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                 {
                   return lhs.response > rhs.response;
                 } );

      std::vector<cv::KeyPoint> anmsPts;

      std::vector<double> radii;
      radii.resize( keypoints.size() );
      std::vector<double> radiiSorted;
      radiiSorted.resize( keypoints.size() );

      const float robustCoeff = 1.11; 

      for( int i = 0; i < keypoints.size(); ++i )
      {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( int j = 0; j < i && keypoints[j].response > response; ++j )
        {
          radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
      }

      std::sort( radiiSorted.begin(), radiiSorted.end(),
                 [&]( const double& lhs, const double& rhs )
                 {
                   return lhs > rhs;
                 } );

      const double decisionRadius = radiiSorted[numToKeep];
      for( int i = 0; i < radii.size(); ++i )
      {
        if( radii[i] >= decisionRadius )
        {
          anmsPts.push_back( keypoints[i] );
        }
      }

      anmsPts.swap( keypoints );
    }



    void drawmatches( const cv::Mat &image0, const cv::Mat &image1, const Features &features0, const Features &features1, const Matches &m01, cv::Mat &imageout )
    {
        cv::addWeighted( image0, 0.5, image1, 0.5, 0, imageout );
        for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
        {
            size_t ind0 = it->first;
            size_t ind1 = it->second;
            
            cv::line( imageout, features0.points[it->first], features1.points[it->second], cv::Scalar(255,255,255) );
        }
    }
    
    Matches filter_matches( const Features &features0, const Features &features1, const Matches &m01, float thresh )
    {
        // filter matches that are longer or shorter than the median  
        // match vector by thresh or more
         
        // get vector lengths
        std::vector<float> lengths;
        for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
        {
            size_t ind0 = it->first;
            size_t ind1 = it->second;
            
            cv::Point2f pt0 = features0.points[it->first];
            cv::Point2f pt1 = features1.points[it->second];
            float dx = pt0.x - pt1.x;
            float dy = pt0.y - pt1.y;
            float length = sqrtf(dx*dx+dy*dy);
            lengths.push_back(length);
        }

        // find median
        std::vector<float> lengths_copy = lengths;
        std::sort(lengths_copy.begin(),lengths_copy.end());
        int mid = lengths_copy.size()/2;
        float median;
        if ( lengths_copy.size() % 2 == 0 ) median = (lengths_copy[mid-1] + lengths_copy[mid])/2;
        else median = lengths_copy[mid];
        
        int n = 0;
        Matches mout;
        for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
        {
            if ( fabsf(lengths[n++] - median) < thresh ) mout[it->first] = it->second;
        }
        return mout;
    }

    DetectorTracker::DetectorTracker( double _min_dist, double _xradius, double _yradius ) :
        min_dist(_min_dist), xradius(_xradius), yradius(_yradius), sift(cv::SIFT::create(20000)) { }
        
    void DetectorTracker::detect( const cv::Mat &image, Features &features )
    {
        assert( image.type() == CV_8UC1 );
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descs;

        int ntokeep = std::max(0,4000 - (int)features.points.size());
        if ( ntokeep == 0 ) return;
        if ( features.empty() )
        {
            sift->detect( image, keypoints );
        } else {
            cv::Mat mask(image.size(),CV_8UC1);
            mask = cv::Scalar(255);
            for ( int i = 0; i < features.points.size(); i++ ) cv::circle(mask,features.points[i],min_dist,cv::Scalar(0),-1);
            sift->detect( image, keypoints, mask );
        }
        
        adaptiveNonMaximalSuppresion(keypoints, ntokeep);
        //std::cout << "after anms, we have " << keypoints.size() << " keypoints\n";
        sift->compute( image, keypoints, descs );
         
        for  ( int i = 0; i < keypoints.size(); i++ )
        {
            features.points.push_back( keypoints[i].pt );
            features.descs.push_back( descs.row(i) );
        }
    }

    void DetectorTracker::track( cv::Mat &image0, cv::Mat &image1,
                const Features &features0, Features &features1,
                Matches &m01 )
    {
        const size_t npts0 = features0.size();
        
        const std::vector<cv::Point2f> &prevPts(features0.points);
        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;
        
        cv::calcOpticalFlowPyrLK(image0, image1, prevPts, nextPts, status, err, cv::Size(xradius*2+1,yradius*2+1));
        
        for ( int i = 0; i < npts0; i++ )
        {
            if ( status[i] == 0 ) continue;
            
            features1.points.push_back( nextPts[i] );
            features1.descs.push_back( features0.descs.row(i) );
            m01[i] = features1.size()-1;
        }
    }

    void match( const Features &features0, const Features &features1, Matches &m01, double ratio )
    {
        cv::BFMatcher matcher;

        std::vector< std::vector<cv::DMatch> > matches;
        
        // p1 = query, p0 = train
        matcher.knnMatch( features1.descs, features0.descs, matches, 2);
        
        for ( int i = 0; i < matches.size(); i++ )
        {
            if ( matches[i][0].distance < ratio*matches[i][1].distance )
            {
                m01[matches[i][0].trainIdx] = matches[i][0].queryIdx;
            }
        }
    }

    void match_points( const cv::Mat &descriptors0, const cv::Mat &descriptors1, Matches &m01, double ratio = 0.75 )
    {
        cv::BFMatcher matcher;

        std::vector< std::vector<cv::DMatch> > matches;
        
        // p1 = query, p0 = train
        matcher.knnMatch( descriptors0, descriptors1, matches, 2);
        
        for ( int i = 0; i < matches.size(); i++ )
        {
            if ( matches[i][0].distance < ratio*matches[i][1].distance )
            {
                m01[matches[i][0].trainIdx] = matches[i][0].queryIdx;
            }
        }
    }

    /*
    void build_feature_tracks( const Intrinsics &intrinsics, const std::string &videopath,
                              std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                              const double inlier_threshold, const double min_rot,
                              const bool inward )
    {
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
        
        std::cout << "image0 has size " << image0.cols << " " << image0.rows << "\n";
        std::cout << "radii: " << xradius << " " << yradius << "\n";
        double min_dist = 20;
        DetectorTracker detectortracker( min_dist, xradius, yradius );

        Features features0;
        detectortracker.detect( image0, features0 );
        
        keyframes.push_back(Keyframe(0,features0));
        image0.copyTo(keyframes[keyframes.size()-1].image);

        std::cout << "detected " << features0.size() << " features in image 0\n";
        
        ransac_lib::LORansacOptions options;
        options.squared_inlier_threshold_ = inlier_threshold*inlier_threshold*Kinv(0)*Kinv(0);
        //options.final_least_squares_ = true;
        options.num_lo_steps_ = 0;
        options.num_lsq_iterations_ = 0;
        options.final_least_squares_ = false;
        ransac_lib::RansacStatistics stats;

        int kf_index = 1;
        int video_index = 1;
        cv::Mat image1_in, image1;
        while ( cap.read( image1_in ) )
        {
            if ( image1_in.channels() == 3 ) cv::cvtColor( image1_in, image1, cv::COLOR_BGR2GRAY );
            else image1 = image1_in;

            Features features1;
            
            Matches m01;
            
            detectortracker.detect( image1, features1 );
            match( features0, features1, m01 );
            std::cout << "tracked " << features1.size() << " features in image " << video_index << "\n";
                
            //cv::Mat matchesim;
            //drawmatches( image0, image1, features0, features1, m01, matchesim );
            //cv::imwrite( "matches" + std::to_string(video_index) + ".png", matchesim );

            std::vector<bool> inliers;
            int ninliers;

            // find inlier matches using relative pose
            RayPairList ray_pair_list;
            ray_pair_list.reserve( m01.size() );
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                size_t index0 = it->first;
                size_t index1 = it->second;

                Eigen::Vector3d loc0 = Eigen::Vector3d( features0.points[index0].x, features0.points[index0].y, 1 );
                Eigen::Vector3d loc1 = Eigen::Vector3d( features1.points[index1].x, features1.points[index1].y, 1 );
                loc0 = Kinv * loc0;
                loc1 = Kinv * loc1;
                
                Ray u;
                u.head(3) = loc0;
                Ray v;
                v.head(3) = loc1;

                ray_pair_list.push_back( std::make_pair( u, v ) );
            }
            SphericalEstimator estimator( ray_pair_list, false, inward );
            
            std::cout << "running ransac on " << ray_pair_list.size() << " matches\n";
            ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,SphericalEstimator> ransac;
            
            Eigen::Matrix3d E;
            ninliers = ransac.EstimateModel( options, estimator, &E, &stats );
            inliers.resize(ray_pair_list.size());
            for ( int i = 0; i < ray_pair_list.size(); i++ )
            {
                inliers[i] = ( estimator.EvaluateModelOnPoint(E,i) < options.squared_inlier_threshold_ );
            }
            fprintf( stdout, "%d: %lu matches and %d inliers (%0.2f%%)\n", video_index, m01.size(), ninliers, (double)ninliers/(double)m01.size()*100 );

            Eigen::Vector3d best_estimator_t;
            Eigen::Vector3d best_estimator_r;
            decompose_spherical_essential_matrix( E, inward, best_estimator_r, best_estimator_t );
            std::cout << best_estimator_r.transpose() << "\n";

            // check for minimum rotation
            if ( best_estimator_r.norm() < min_rot * M_PI / 180. ) {
                std::cout << "skipping frame\n";
                video_index++;
                continue;
            }

            Matches m01inliers;
            size_t i = 0;
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                if ( inliers[i++] ) m01inliers[it->first] = it->second;
            }
            //cv::Mat inliersim;
            //drawmatches( image0, image1, features0, features1, m01inliers, inliersim );
            //cv::imwrite( "inliers" + std::to_string(video_index) + ".png", inliersim );
            
            image1.copyTo(image0);
            features0 = features1;
            
            std::cout << "features0 now has " << features0.size() << " features\n";
            // also detect new points that are some distance away from matched points

            keyframes.push_back(Keyframe(video_index,features0));
            image0.copyTo(keyframes[keyframes.size()-1].image);
            image_matches.push_back( ImageMatch( kf_index-1, kf_index, m01, so3exp(best_estimator_r) ) );
            video_index++;
            kf_index++;
        }
    }
    */

    void detect_features( const std::string &videopath, std::vector<Keyframe> &keyframes )
    {
        cv::VideoCapture cap(videopath);
        
        int i = 0;
        cv::Mat image_in, image;
        while ( cap.read(image_in) )
        {
            if ( image_in.channels() == 3 ) cv::cvtColor( image_in, image, cv::COLOR_BGR2GRAY );
            else image = image_in;
            Features features;
            keyframes.push_back(Keyframe(i,"",features));
            image.copyTo(keyframes[keyframes.size()-1].image);
            i++;
        }
    
        if ( keyframes.empty() )
        {
            std::cout << "error: could not read single frame from " << videopath << "\n";
            exit(1);
        }

#pragma omp parallel for
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &keyframe = keyframes[i];
            DetectorTracker detector;
            detector.detect( keyframe.image, keyframe.features );
            //std::cout << "image " << i << " has " << keyframe.features.size() << " features\n";
        }
    }
    
    int estimate_pairwise( const Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const bool inward, std::vector<ImageMatch> &image_matches_out )
    {
        Eigen::Matrix3d Kinv = intrinsics.getKinv();
        
        ransac_lib::LORansacOptions options;
        options.squared_inlier_threshold_ = inlier_threshold*inlier_threshold*Kinv(0)*Kinv(0);
        options.num_lo_steps_ = 0;
        options.num_lsq_iterations_ = 0;
        options.final_least_squares_ = true;

        std::vector<ImageMatch> my_image_matches; 
        for ( int index0 = 0; index0 < keyframes.size(); index0++ )
        {
            for ( int index1 = index0+1; index1 < keyframes.size(); index1++ )
            {
                Matches m;
                Eigen::Matrix3d R;
                ImageMatch image_match(index0, index1, m, R);
                my_image_matches.push_back(image_match);
            }
        }

#pragma omp parallel for
        for ( int i = 0; i < my_image_matches.size(); i++ ) 
        {
            ImageMatch &image_match = my_image_matches[i];
            int index0 = image_match.index0;
            int index1 = image_match.index1;
            const Features &features0 = keyframes[index0].features;
            const Features &features1 = keyframes[index1].features;

            Matches m01;
             
            for ( auto m : image_matches )
            { 
                if ( m.index0 == index0 && m.index1 == index1 )
                {
                    m01 = m.matches;
                    break;
                }
            }
            if ( m01.size() < min_num_inliers ) continue;

            std::vector<bool> inliers;
            int ninliers;

            // find inlier matches using relative pose
            RayPairList ray_pair_list;
            ray_pair_list.reserve( m01.size() );
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                cv::Point2f pt0 = features0.points[it->first];
                cv::Point2f pt1 = features1.points[it->second];

                Eigen::Vector3d loc0 = Eigen::Vector3d( pt0.x, pt0.y, 1 );
                Eigen::Vector3d loc1 = Eigen::Vector3d( pt1.x, pt1.y, 1 );
                
                loc0 = Kinv * loc0;
                loc1 = Kinv * loc1;
                
                Ray u;
                u.head(3) = loc0;
                Ray v;
                v.head(3) = loc1;

                ray_pair_list.push_back( std::make_pair( u, v ) );
            }
                
            SphericalEstimator estimator(ray_pair_list, false, inward);
            
            ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,SphericalEstimator> ransac;
            ransac_lib::RansacStatistics stats;
            Eigen::Matrix3d E;
            if ( m01.empty() )
            {
                ninliers = 0;
            } else {
                ninliers = ransac.EstimateModel( options, estimator, &E, &stats );
                inliers.resize(ray_pair_list.size());
                for ( int i = 0; i < ray_pair_list.size(); i++ )
                {
                    inliers[i] = ( estimator.EvaluateModelOnPoint(E,i) < options.squared_inlier_threshold_ );
                }
            }
            //fprintf( stdout, "%d %d: %lu matches and %d inliers (%0.2f%%)\n",
                     //keyframes[index0].index, keyframes[index1].index, m01.size(), ninliers, (double)ninliers/(double)m01.size()*100 );

            Matches m01inliers;
            size_t j = 0;
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                if ( inliers[j++] ) m01inliers[it->first] = it->second;
            }
                
            /*
            cv::Mat matchesim;
            drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01inliers, matchesim );
            cv::imwrite( "inliers" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", matchesim );
            */

            if ( ninliers > min_num_inliers )
            {
                Eigen::Vector3d best_estimator_t;
                Eigen::Vector3d best_estimator_r;
                decompose_spherical_essential_matrix( E, inward, best_estimator_r, best_estimator_t );
                //std::cout << best_estimator_r.transpose() << "\n";

                image_match.matches = m01inliers;
                image_match.R = so3exp(best_estimator_r);
            }
        }
        
        int loop_closure_count = 0;
        for ( int i = 0; i < my_image_matches.size(); i++ )
        {
            if ( my_image_matches[i].matches.empty() ) continue;
            if ( my_image_matches[i].index0 + 1 != my_image_matches[i].index1 ) loop_closure_count++;
            image_matches_out.push_back( my_image_matches[i] );
        }
        
        return loop_closure_count;
    }

    void match_exhaustive( const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches )
    {
        for ( int index0 = 0; index0 < keyframes.size(); index0++ )
        {
            for ( int index1 = index0+1; index1 < keyframes.size(); index1++ )
            {
                Matches m;
                Eigen::Matrix3d R;
                ImageMatch image_match(index0, index1, m, R);
                image_matches.push_back(image_match);
            }
        }

#pragma omp parallel for
        for ( int i = 0; i < image_matches.size(); i++ ) 
        {
            //if ( i % 10 == 0 ) std::cout << i+1 << " / " << image_matches.size() << "\n";
            ImageMatch &image_match = image_matches[i];
            int index0 = image_match.index0;
            int index1 = image_match.index1;
            const Features &features0 = keyframes[index0].features;
            const Features &features1 = keyframes[index1].features;

            match( features0, features1, image_match.matches );
        }
    }

    int make_loop_closures( const Intrinsics &intrinsics, const std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches,
                            const double inlier_threshold, const int min_num_inliers, const int num_frames_begin, const int num_frames_end, const bool best_only, const bool inward )
    {
        int width = intrinsics.centerx*2;
        int height = intrinsics.centery*2;
        float thresh = 0.02*sqrtf(width*width+height*height);
        Eigen::Matrix3d Kinv = intrinsics.getKinv();
        
        ransac_lib::LORansacOptions options;
        options.squared_inlier_threshold_ = inlier_threshold*inlier_threshold*Kinv(0)*Kinv(0);
        //options.final_least_squares_ = true;
        options.num_lo_steps_ = 0;
        options.num_lsq_iterations_ = 0;
        options.final_least_squares_ = false;
        ransac_lib::RansacStatistics stats;

        ImageMatch best_image_match( -1, -1, Matches(), Eigen::Matrix3d::Identity() );
        int loop_closure_count = 0;

        DetectorTracker detectortracker( 0, 0, 0 );
                
        int count0 = 0;
        for ( int index0 = 0; index0 < keyframes.size(); index0++ )
        {
            if ( ++count0 > num_frames_begin ) break;

            const Features &features0 = keyframes[index0].features;
            
            int count1 = 0;
            for ( int index1 = keyframes.size()-1; index1 >= index0+1; index1-- )
            {
                if ( ++count1 > num_frames_end ) break;

                if ( index1 == index0 + 1 || index0 == index1 + 1 ) continue;
                            
                const Features &features1 = keyframes[index1].features;

                Matches m01;
                
                match( features0, features1, m01 );
                if ( m01.size() < min_num_inliers ) continue;

                //cv::Mat matchesim;
                //drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01, matchesim );
                //cv::imwrite( "matches" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", matchesim );

                std::vector<bool> inliers;
                int ninliers;

                // find inlier matches using relative pose
                RayPairList ray_pair_list;
                ray_pair_list.reserve( m01.size() );
                for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
                {
                    cv::Point2f pt0 = features0.points[it->first];
                    cv::Point2f pt1 = features1.points[it->second];

                    Eigen::Vector3d loc0 = Eigen::Vector3d( pt0.x, pt0.y, 1 );
                    Eigen::Vector3d loc1 = Eigen::Vector3d( pt1.x, pt1.y, 1 );
                    
                    loc0 = Kinv * loc0;
                    loc1 = Kinv * loc1;
                    
                    Ray u;
                    u.head(3) = loc0;
                    Ray v;
                    v.head(3) = loc1;

                    ray_pair_list.push_back( std::make_pair( u, v ) );
                }
                
                SphericalEstimator estimator(ray_pair_list,false,inward);
                
                ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,SphericalEstimator> ransac;
                ransac_lib::RansacStatistics stats;
                Eigen::Matrix3d E;
                if ( m01.empty() )
                {
                    ninliers = 0;
                } else {
                    ninliers = ransac.EstimateModel( options, estimator, &E, &stats );
                    inliers.resize(ray_pair_list.size());
                    for ( int i = 0; i < ray_pair_list.size(); i++ )
                    {
                        inliers[i] = ( estimator.EvaluateModelOnPoint(E,i) < options.squared_inlier_threshold_ );
                    }
                }
                fprintf( stdout, "%d %d: %lu matches and %d inliers (%0.2f%%)\n",
                        keyframes[index0].index, keyframes[index1].index, m01.size(), ninliers, (double)ninliers/(double)m01.size()*100 );

                Matches m01inliers;
                size_t i = 0;
                for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
                {
                    if ( inliers[i++] ) m01inliers[it->first] = it->second;
                }

                if ( ninliers > min_num_inliers )
                {
                    Eigen::Vector3d best_estimator_t;
                    Eigen::Vector3d best_estimator_r;
                    decompose_spherical_essential_matrix( E, inward, best_estimator_r, best_estimator_t );
                    std::cout << best_estimator_r.transpose() << "\n";

                    /*
                    cv::Mat inliersim;
                    drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01inliers, inliersim );
                    cv::imwrite( "loop" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", inliersim );
                    */

                    //m01inliers = filter_matches(features0,features1,m01inliers,thresh);

                    //drawmatches( keyframes[index0].image, keyframes[index1].image, features0, features1, m01inliers, inliersim );
                    //cv::imwrite( "inliers_after" + std::to_string(keyframes[index0].index) + "-" + std::to_string(keyframes[index1].index) + ".png", inliersim );

                    ImageMatch image_match(index0, index1, m01inliers, so3exp(best_estimator_r));
                    if ( ninliers > best_image_match.matches.size() ) best_image_match = image_match;
                    if ( !best_only )
                    {
                        image_matches.push_back(image_match);
                        loop_closure_count++;
                    }
                }
            }
        }
        if ( best_only )
        {
            if ( best_image_match.matches.empty() ) return 0;
            image_matches.push_back(best_image_match);
            return 1;
        }
        return loop_closure_count;
    }
    
    void find_largest_connected_component( std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches )
    {
        using namespace boost;
        {
            typedef adjacency_list <vecS, vecS, undirectedS> Graph;

            Graph G;
            for ( auto im : image_matches )
            {
                add_edge( im.index0, im.index1, G );
            }
    
            std::vector<int> component(num_vertices(G));
            int num = connected_components(G, &component[0]);
    
            std::vector<int> sizes(num);
            for ( int i = 0; i < num; i++ ) sizes[i] = 0;
            for ( int i = 0; i < num_vertices(G); i++ ) sizes[component[i]]++;
            for ( int i = 0; i < num; i++ ) std::cout << "component " << i << " has " << sizes[i] << "\n";
            
            int best_size = sizes[0];
            int best_index = 0;
            for ( int i = 1; i < num; i++ ) 
            {
                if ( sizes[i] > best_size )
                {
                    best_index = i;
                    best_size = sizes[i];
                }
            }
            
            std::cout << "largest connected component has " << best_size << " vertices\n";
            
            std::map<int,int> old2new;
            std::vector<Keyframe> new_keyframes;
            std::vector<ImageMatch> new_image_matches;
            for ( int i = 0; i < keyframes.size(); i++ )
            {
                if ( component[i] == best_index )
                {
                    old2new[i] = new_keyframes.size();
                    new_keyframes.push_back(keyframes[i]);
                }
            }
            for ( int i = 0; i < image_matches.size(); i++ )
            {
                if ( component[image_matches[i].index0] == best_index && component[image_matches[i].index1] == best_index )
                {
                    image_matches[i].index0 = old2new[image_matches[i].index0];
                    image_matches[i].index1 = old2new[image_matches[i].index1];
                    new_image_matches.push_back(image_matches[i]);
                }
            }
            keyframes = new_keyframes;
            image_matches = new_image_matches;
        }
    }

    void initialize_rotations_sequential( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations )
    {
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
    }

    void initialize_rotations_gopt( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations )
    {
        rotations.resize(num_cameras);
        for ( int i = 0 ; i < num_cameras; i++ ) rotations[i] = Eigen::Matrix3d::Identity();
        
        gopt::graph::ViewGraph view_graph;
        for ( int i = 0 ; i < num_cameras; i++ ) {
            view_graph.AddNode(gopt::graph::ViewNode(i));
        }
        for ( auto im : image_matches ) {
            gopt::graph::ViewEdge edge(im.index0,im.index1);
            edge.rel_rotation = so3ln(im.R);
            view_graph.AddEdge(edge);
        }
        
        /*
        std::unordered_map<gopt::ImagePair, gopt::TwoViewGeometry> view_pairs;
        for ( auto im : image_matches ) {
            view_pairs[gopt::ImagePair(im.index0,im.index1)].rel_rotation = so3ln(im.R);
        }
        */
        std::unordered_map<gopt::image_t, Eigen::Vector3d> global_rotations;
        //rotest.EstimateRotations(view_pairs, &global_rotations);
        gopt::RotationEstimatorOptions options;
        options.verbose = false;
        options.Setup();
        view_graph.RotationAveraging(options, &global_rotations);
        for ( int index = 0; index < num_cameras; index++ )
        {
            rotations[index] = so3exp(global_rotations[index]);
            //std::cout << index << "\t" << global_rotations[index].transpose() << "\n";
        }
    }

    double refine_rotations( const int num_cameras, const std::vector<ImageMatch> &image_matches, std::vector<Eigen::Matrix3d> &rotations )
    {
        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches[i].index0, image_matches[i].index1, image_matches[i].R ) );
        }
        
        return optimize_rotations( rotations, relative_rotations );
    }

    void build_sfm( std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches, const std::vector<Eigen::Matrix3d> &rotations,
                   sphericalsfm::SfM &sfm, bool spherical, bool merge, bool inward, int fix_camera )
    {
        std::cout << "building tracks\n";
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            keyframes[i].features.tracks.resize( keyframes[i].features.size() );
            for ( int j = 0; j < keyframes[i].features.size(); j++ )
            {
                keyframes[i].features.tracks[j] = -1;
            }
        }
        
        // add cameras
        std::cout << "adding cameras\n";
        for ( int index = 0; index < keyframes.size(); index++ )
        {
            Eigen::Vector3d t(0,0,-1);
            if ( inward ) t[2] = 1;
            int camera = sfm.AddCamera( Pose( t, so3ln(rotations[index]) ), keyframes[index].name );
            sfm.SetRotationFixed( camera, (index==fix_camera) );
            sfm.SetTranslationFixed( camera, spherical ? true : (index==fix_camera) );
        }

        std::cout << "adding tracks\n";
        std::cout << "number of keyframes is " << keyframes.size() << "\n";
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            const Matches &m01 = image_matches[i].matches;
            
            const int index0 = image_matches[i].index0;
            const int index1 = image_matches[i].index1;
            //std::cout << "match between keyframes " << index0 << " and " << index1 << " has " << m01.size() << " matches\n";
            
            const Features &features0 = keyframes[index0].features;
            const Features &features1 = keyframes[index1].features;
            
            for ( Matches::const_iterator it = m01.begin(); it != m01.end(); it++ )
            {
                const cv::Point2f pt0 = features0.points[it->first];
                const cv::Point2f pt1 = features1.points[it->second];
                
                int &track0 = keyframes[index0].features.tracks[it->first];
                int &track1 = keyframes[index1].features.tracks[it->second];
                
                Observation obs0( pt0.x-sfm.GetIntrinsics().centerx, pt0.y-sfm.GetIntrinsics().centery );
                Observation obs1( pt1.x-sfm.GetIntrinsics().centerx, pt1.y-sfm.GetIntrinsics().centery );

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
                    track0 = track1 = sfm.AddPoint( Eigen::Vector3d::Zero(), features0.descs.row(it->first) );

                    sfm.SetPointFixed( track0, false );
                    
                    sfm.AddObservation( index0, track0, obs0 );
                    sfm.AddObservation( index1, track1, obs1 );
                }
                else if ( track0 != -1 && track1 != -1 && ( track0 != track1 ) )
                {
                    if ( merge )
                    {
                        // track1 will be removed
                        sfm.MergePoint( track0, track1 );
                        
                        // update all features with track1 and set to track0
                        for ( int i = 0; i < keyframes.size(); i++ )
                        {
                            for ( int j = 0; j < keyframes[i].features.size(); j++ )
                            {
                                if ( keyframes[i].features.tracks[j] == track1 ) keyframes[i].features.tracks[j] = track0;
                            }
                        }
                    } else {
                        sfm.AddObservation( index0, track0, obs0 );
                        sfm.AddObservation( index1, track1, obs1 );
                    }
                }
            }
        }

        std::cout << "retriangulating...\n";
        sfm.Retriangulate();

    }

    void show_reprojection_error( std::vector<Keyframe> &keyframes, SfM &sfm )
    {
        double focal = sfm.GetIntrinsics().focal;
        double centerx = sfm.GetIntrinsics().centerx;
        double centery = sfm.GetIntrinsics().centery;
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            cv::Mat image;
            if ( keyframes[i].image.channels() == 1 )
            {
                cv::cvtColor( keyframes[i].image, image, cv::COLOR_GRAY2BGR );
            } else {    
                keyframes[i].image.copyTo(image);
            }
            
            
            Pose pose = sfm.GetPose( i );
            
            // draw points and observations
            for ( int point_index = 0; point_index < sfm.GetNumPoints(); point_index++ )
            {
                // get observation
                Observation obs;
                if ( !sfm.GetObservation( i, point_index, obs ) ) continue;
                
                // get point
                Point point = sfm.GetPoint( point_index );
                if ( point.norm() == 0 ) continue;
                
                Eigen::Vector3d proj3 = pose.apply( point );
                Eigen::Vector2d proj = focal*proj3.head(2)/proj3(2);
                
                // check re-proj error
                Eigen::Vector2d diff = proj - obs;
                double dist = diff.norm();
                
                dist = std::min(dist,10.);
                cv::Scalar color = cv::Scalar(0,(1.-dist/10.)*255,255);
                //if ( dist > 2. ) color = cv::Scalar(0,0,255);
                
                cv::Point2f loc(obs[0]+centerx,obs[1]+centery);
                cv::circle(image, loc, 3, color,  -1 );
            }
            
            char imagepath[1024];
            sprintf(imagepath, "reproj%06d.jpg", i );
            cv::imwrite(imagepath, image);
        }
    }
    
    struct FocalLengthSearchParams
    {
        int num_cameras;
        std::vector<Eigen::Matrix3d> &Es;
        std::vector<ImageMatch> &image_matches;
        const bool inward;
        const bool sequential;
        const double focal_guess;
        std::vector<Eigen::Matrix3d> rotations;
        FocalLengthSearchParams(
            int _num_cameras,
            std::vector<Eigen::Matrix3d> &_Es,
            std::vector<ImageMatch> &_image_matches,
            const bool _inward,
            const bool _sequential,
            const double _focal_guess
        ) : num_cameras(_num_cameras),
            Es(_Es),
            image_matches(_image_matches),
            inward(_inward),
            sequential(_sequential),
            focal_guess(_focal_guess) { }
    };
    
    std::vector<ImageMatch> filter_image_matches( std::vector<ImageMatch> &image_matches, double err_thresh_rad )
    {
        std::vector<bool> good(image_matches.size());
        for ( int i = 0; i < good.size(); i++ ) good[i] = false;

        FILE *f = fopen("filter.txt","w");
        int triplet_count = 0;
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            const ImageMatch &im0 = image_matches[i];
            Eigen::Matrix3d Rij = im0.R;
            for ( int j = 0; j < image_matches.size(); j++ )
            {
                const ImageMatch &im1 = image_matches[j];
                if ( im1.index0 != im0.index1 ) continue;

                Eigen::Matrix3d Rjk = im1.R;
                for ( int k = 0; k < image_matches.size(); k++ )
                {
                    const ImageMatch &im2 = image_matches[k];
                    if ( im2.index0 != im0.index0 ) continue;
                    if ( im2.index1 != im1.index1 ) continue;
                    Eigen::Matrix3d Rik = im2.R;
                    
                    double err = so3ln(Rij*Rjk*Rik.transpose()).norm();
                    std::cout << "triplet " << triplet_count << ": " << im0.index0 << " " << im0.index1 << " " << im1.index1 << "\t" << err*180/M_PI << "\n";
                    fprintf(f,"%d %d %d %f\n",im0.index0,im0.index1,im1.index1,err*180/M_PI);

                    if ( err < err_thresh_rad )
                    {
                        good[i] = true;
                        good[j] = true;
                        good[k] = true;
                    }
                    triplet_count++;
                }
            }
        }
        fclose(f);

        int count = 0;
        std::vector<ImageMatch> image_matches_new;
        for ( int i = 0; i < good.size(); i++ ) {
            if ( good[i] ) {
                count++;
                image_matches_new.push_back(image_matches[i]);
            }
        }
        std::cout << count << " / " << good.size() << " good edges\n";
        
        return image_matches_new;
    }

    static double total_rotation_cost_fn( double focal, void * params )
    {
        FocalLengthSearchParams *search_params = (FocalLengthSearchParams *)params;

        std::vector<ImageMatch> image_matches_new(search_params->image_matches);

        Eigen::Vector3d T(focal/search_params->focal_guess,focal/search_params->focal_guess,1);
        for ( int i = 0; i < search_params->image_matches.size(); i++ )
        {
            Eigen::Matrix3d E_new = T.asDiagonal() * search_params->Es[i] * T.asDiagonal();
            Eigen::Vector3d r_new, t_new;
            decompose_spherical_essential_matrix( E_new, search_params->inward, r_new, t_new );
            image_matches_new[i].R = so3exp(r_new);
        }

        if ( search_params->sequential )
        {
            initialize_rotations_sequential( search_params->num_cameras, image_matches_new, search_params->rotations );
        } else {
            initialize_rotations_gopt( search_params->num_cameras, image_matches_new, search_params->rotations );
        }

        double total_rot = 0;
        for ( int i = 1; i < search_params->num_cameras; i++ )
        {
            Eigen::Matrix3d relative_rotation = search_params->rotations[i].transpose() * search_params->rotations[i-1];
            total_rot += so3ln(relative_rotation).norm();
        }
        
        double cost = fabs(2*M_PI-total_rot);

        return cost;
    }

    static void transform_image_matches( double focal, const FocalLengthSearchParams *search_params, std::vector<ImageMatch> &image_matches_new )
    {
        image_matches_new.clear();

        Eigen::Vector3d T(focal/search_params->focal_guess,focal/search_params->focal_guess,1);
        for ( int i = 0; i < search_params->image_matches.size(); i++ )
        {
            ImageMatch &im = search_params->image_matches[i];
            Eigen::Matrix3d E_new = T.asDiagonal() * search_params->Es[i] * T.asDiagonal();
            Eigen::Vector3d r_new, t_new;
            decompose_spherical_essential_matrix( E_new, search_params->inward, r_new, t_new );
            image_matches_new.push_back(ImageMatch(im.index0,im.index1,im.matches,so3exp(r_new)));
        }
    }
    
    static void initialize_rotations( int num_cameras, const std::vector<ImageMatch> &image_matches, bool sequential, std::vector<Eigen::Matrix3d> &rotations )
    {
        if ( sequential ) {
            initialize_rotations_sequential( num_cameras, image_matches, rotations );
        } else {
            initialize_rotations_gopt( num_cameras, image_matches, rotations );
        }
    }

    static double loop_constraint_cost_fn( double focal, void * params )
    {
        const FocalLengthSearchParams *search_params = (FocalLengthSearchParams *)params;
        std::vector<ImageMatch> image_matches_new;
        std::vector<Eigen::Matrix3d> rotations;

        transform_image_matches( focal, search_params, image_matches_new );
        initialize_rotations( search_params->num_cameras, image_matches_new, search_params->sequential, rotations );

        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches_new.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches_new[i].index0, image_matches_new[i].index1, image_matches_new[i].R ) );
        }
        double cost = get_cost(rotations, relative_rotations );

        return cost;
    }

    static double run_optimization( double &focal, void * params, double min_focal, double max_focal )
    {
        FocalLengthSearchParams *search_params = (FocalLengthSearchParams *)params;

        std::cout << "got here\n";
        std::vector<ImageMatch> image_matches_new(search_params->image_matches);
        std::cout << "got here\n";

        Eigen::Vector3d T(focal/search_params->focal_guess,focal/search_params->focal_guess,1);
        std::cout << "got here\n";
        for ( int i = 0; i < search_params->image_matches.size(); i++ )
        {
            Eigen::Matrix3d E_new = T.asDiagonal() * search_params->Es[i] * T.asDiagonal();
            Eigen::Vector3d r_new, t_new;
            decompose_spherical_essential_matrix( E_new, search_params->inward, r_new, t_new );
            image_matches_new[i].R = so3exp(r_new);
        }
        std::cout << "got here\n";

        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches_new.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches_new[i].index0, image_matches_new[i].index1, image_matches_new[i].R ) );
        }
        std::cout << "starting optimization\n";
        double cost = optimize_rotations_and_focal_length( search_params->rotations, relative_rotations, focal, min_focal, max_focal, search_params->focal_guess, search_params->inward );
        
        return cost;
    }

    double optimize_focal_length( double (* function) (double x, void * params), double initial_focal, double min_focal, double max_focal, FocalLengthSearchParams *params )
    {
        int status;
        int iter = 0, max_iter = 100;
        const gsl_min_fminimizer_type *T;
        gsl_min_fminimizer *s;
        gsl_function F;

        F.function = function;
        F.params = params;
        
        double m = initial_focal;
        double a = min_focal;
        double b = max_focal;
        
        std::cout << a << " " << m << " " << b << "\n";
        std::cout << function(a,params) << " " << function(m,params) << " " << function(b,params) << "\n";

        T = gsl_min_fminimizer_brent;
        s = gsl_min_fminimizer_alloc (T);
        gsl_min_fminimizer_set( s, &F, m, a, b );

        printf ("using %s method\n",
          gsl_min_fminimizer_name (s));

        printf ("%5s [%9s, %9s] %9s %10s %9s\n",
          "iter", "lower", "upper", "min",
          "err", "err(est)");

        printf ("%5d [%.7f, %.7f] %.7f %+.7f\n",
          iter, a, b,
          m, b - a);

      do
        {
          iter++;
          status = gsl_min_fminimizer_iterate (s);

          m = gsl_min_fminimizer_x_minimum (s);
          a = gsl_min_fminimizer_x_lower (s);
          b = gsl_min_fminimizer_x_upper (s);

          status
            = gsl_min_test_interval (a, b, 0.001, 0.0);

          if (status == GSL_SUCCESS)
            printf ("Converged:\n");

          printf ("%5d [%.7f, %.7f] "
                  "%.7f %.7f\n",
                  iter, a, b,
                  m, b - a);
        }
      while (status == GSL_CONTINUE && iter < max_iter);

      gsl_min_fminimizer_free (s);
      return m;
    }

    bool find_best_focal_length_opt( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal )
    {
        std::vector<Eigen::Matrix3d> Es(image_matches.size());
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            make_spherical_essential_matrix(image_matches[i].R,inward,Es[i]);
        }
        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches[i].index0, image_matches[i].index1, image_matches[i].R ) );
        }

        FocalLengthSearchParams params(
            num_cameras,
            Es,
            image_matches,
            inward,
            sequential,
            focal_guess );
        
        std::cout << "focal guess: " << focal_guess << "\n";
        
        std::cout << "focal min, guess, max: " << min_focal << " " << focal_guess << " " << max_focal << "\n";
        double focal_init = focal_guess;
        double lower_cost = loop_constraint_cost_fn( min_focal, &params );
        double mid_cost = loop_constraint_cost_fn( focal_init, &params );
        double upper_cost = loop_constraint_cost_fn( max_focal, &params );
        std::cout << "original costs for focal_min, focal_init, focal_max:"<< "\n";
        std::cout << lower_cost << "\t" << mid_cost << "\t" << upper_cost << "\n";
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(min_focal,max_focal);
        int num_trials = 0;
        while ( (lower_cost < mid_cost) || (upper_cost < mid_cost) )
        {
            if ( num_trials++ > 100 )
            {
                std::cout << "ERROR: could not find suitable focal length initialzation. Try increasing focal bounds.\n";
                return false;
            }
            focal_init = dist(gen);
            mid_cost = loop_constraint_cost_fn( focal_init, &params );
        }
        std::cout << "new costs for focal_min, focal_init, focal_max:"<< "\n";
        std::cout << lower_cost << "\t" << mid_cost << "\t" << upper_cost << "\n";
        best_focal = optimize_focal_length( &loop_constraint_cost_fn, focal_init, min_focal, max_focal, &params );
        std::cout << "best focal: " << best_focal << "\n";

        run_optimization( best_focal, &params, min_focal, max_focal );
        rotations = params.rotations;
        std::cout << "focal after optimization: " << best_focal << "\n";

        return true;//(total_rot >= min_total_rot && total_rot <= max_total_rot);
    }

    bool find_best_focal_length_grid( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 int num_steps,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal )
    {
        std::vector<Eigen::Matrix3d> Es(image_matches.size());
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            make_spherical_essential_matrix(image_matches[i].R,inward,Es[i]);
        }
        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches[i].index0, image_matches[i].index1, image_matches[i].R ) );
        }

        FocalLengthSearchParams params(
            num_cameras,
            Es,
            image_matches,
            inward,
            sequential,
            focal_guess );
        
        bool found_one = false;
        double best_cost = INFINITY;
        best_focal = focal_guess;
    
        double min_good_focal = INFINITY;
        double max_good_focal = 0;
        
        FILE *f = fopen("costs.txt","w");
        double focal_step = (max_focal-min_focal)/(num_steps-1);
        for ( int step = 0; step < num_steps; step++ )
        {
            double focal = min_focal + focal_step * step;

            double cost = loop_constraint_cost_fn( focal, &params );
            
            // find total rotation
            double total_rot = 0;
            for ( int i = 1; i < num_cameras; i++ )
            {
                Eigen::Matrix3d relative_rotation = params.rotations[i].transpose() * params.rotations[i-1];
                total_rot += so3ln(relative_rotation).norm();
            }
            std::cout << focal << " " << cost << " " << total_rot << "\n";
            fprintf(f,"%d %lf %lf %lf\n",step,focal,cost,total_rot);
            //if ( total_rot > min_total_rot && total_rot < max_total_rot )
            //{
                //if ( focal < min_good_focal ) min_good_focal = focal;
                //if ( focal > max_good_focal ) max_good_focal = focal;
            //}
                
            if ( cost < best_cost ) //&& 
                //total_rot > min_total_rot &&
                //total_rot < max_total_rot )
            {
                best_cost = cost;
                best_focal = focal;
                rotations = params.rotations;
                found_one = true;
            }
            
            //if ( total_rot > max_total_rot ) break;
            
            char path[1024];
            sprintf(path,"centers%03d.obj",step);
            FILE *objf = fopen(path,"w");
            for ( int i = 0; i < params.rotations.size(); i++ )
            {
                const Eigen::Vector3d t(0,0,-1);
                const Eigen::Vector3d c = -params.rotations[i].transpose() * t;
                fprintf(objf,"v %lf %lf %lf\n",c(0),c(1),c(2));
            }
            fclose(objf);
        }
        fclose(f);

        if ( found_one )
        {
            std::cout << "before optimization: " << best_focal << "\n";
            params.rotations = rotations;
            run_optimization( best_focal, &params, min_focal, max_focal );
            rotations = params.rotations;
            std::cout << "after optimization: " << best_focal << "\n";
        }
        
        return found_one;
    }

    bool find_best_focal_length_random( int num_cameras,
                                 std::vector<ImageMatch> &image_matches,
                                 bool inward,
                                 bool sequential,
                                 double focal_guess,
                                 double min_focal,
                                 double max_focal,
                                 int num_trials,
                                 std::vector<Eigen::Matrix3d> &rotations,
                                 double &best_focal )
    {
        std::vector<Eigen::Matrix3d> Es(image_matches.size());
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            make_spherical_essential_matrix(image_matches[i].R,inward,Es[i]);
        }
        std::vector<RelativeRotation> relative_rotations;
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            relative_rotations.push_back( RelativeRotation( image_matches[i].index0, image_matches[i].index1, image_matches[i].R ) );
        }

        FocalLengthSearchParams params(
            num_cameras,
            Es,
            image_matches,
            inward,
            sequential,
            focal_guess );
        
        double best_cost = INFINITY;
        best_focal = focal_guess;
    
        double min_good_focal = INFINITY;
        double max_good_focal = 0;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(min_focal,max_focal);

        std::vector<double> focals(num_trials);
        std::vector<double> costs(num_trials);
#pragma omp parallel for
        for ( int trial = 0; trial < num_trials; trial++ )
        {
            double focal = dist(gen);
            double cost = loop_constraint_cost_fn( focal, &params );
            focals[trial] = focal;
            costs[trial] = cost;
        }
        for ( int trial = 0; trial < num_trials; trial++ )
        {
            if ( costs[trial] < best_cost ) 
            {
                best_cost = costs[trial];
                best_focal = focals[trial];
            }
        }

        std::vector<ImageMatch> image_matches_new;

        transform_image_matches( best_focal, &params, image_matches_new );
        initialize_rotations( num_cameras, image_matches_new, sequential, rotations );
         
        std::cout << "before optimization: " << best_focal << "\n";
        params.rotations = rotations;
        run_optimization( best_focal, &params, min_focal, max_focal );
        rotations = params.rotations;
        std::cout << "after optimization: " << best_focal << "\n";
        
        return true;
    }

}

