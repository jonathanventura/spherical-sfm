#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>

#include <opencv2/cudaoptflow.hpp>
#include "opencv2/cudaarithm.hpp"

#include <cstdio>
#include <iostream>

#include <sphericalsfm/so3.h>
#include <sphericalsfm/plane_estimator.h>
#include <sphericalsfm/preemptive_ransac.h>

#include "stereo_panorama_tools.h"

using namespace sphericalsfm;

namespace stereopanotools {

    static cv::Ptr<cv::cuda::BroxOpticalFlow> optflow = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 150, 10);

    static const double depth = 10; // depth of plane
    static const double synth_radius = 0.5; // radius of circle of synthetic views
    static const double synth_focal_factor = 1.2; // synthetic focal = factor * input focal

    void _compute_flow( cv::Mat &prev, cv::Mat &next, cv::Mat &out, bool upsample = false )
    {
        cv::cuda::GpuMat d_frame0(prev);
        cv::cuda::GpuMat d_frame1(next);

        cv::cuda::GpuMat d_flow(d_frame0.size(), CV_32FC2), d_flowxy;
        cv::cuda::GpuMat d_frame0f;
        cv::cuda::GpuMat d_frame1f;

        d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
        d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

        optflow->calc(d_frame0f, d_frame1f, d_flow);
        
        d_flow.download(out);

        if ( upsample )
        {
            cv::Mat out2;
            cv::resize(out, out2, cv::Size(0,0), 2, 2, cv::INTER_LINEAR);
            out2 *= 2;
            out = out2;
        }
    }

    Eigen::Vector3d project(const Eigen::Vector3d &vec, const Eigen::Vector3d &up)
    {
      return vec - up * (vec.dot(up));
    }

    double signed_angle_between( const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &up )
    {
        return atan2(a.cross(b).dot(up),a.dot(b));
    }

    bool get_synthetic_column_maps( 
                                    double focal, double centerx, double centery,
                                    double theta, double phi,
                                    const Keyframe &left, const Keyframe &right,
                                    cv::Mat &xL, cv::Mat &xR )
    {
        Eigen::Vector3d synth_t(0,0,-synth_radius);
        Eigen::Matrix3d synth_R = so3exp(Eigen::Vector3d(0,-theta,0));
        Eigen::Vector3d synth_center = -synth_R.transpose() * synth_t;
        double synth_focal = focal * synth_focal_factor;
        
        Eigen::Vector3d left_center = -left.R.transpose() * left.t;
        Eigen::Vector3d right_center = -right.R.transpose() * right.t;

        for ( int y = 0; y < xL.rows; y++ )
        {
          double col = tan(phi);

          Eigen::Vector3d synth_x(col,(y-centery)/synth_focal,1);
          Eigen::Vector3d synth_X = synth_x * depth;
          Eigen::Vector3d world_X = synth_R.transpose() * (synth_X - synth_t);
                
          // find projection in left and right images
          Eigen::Vector3d XL = left.R * world_X + left.t;
          Eigen::Vector3d XR = right.R * world_X + right.t;
          
          if ( XL(2) <= 0 ) return false;
          if ( XR(2) <= 0 ) return false;
          
          Eigen::Vector2d XL_proj(focal*XL(0)/XL(2)+centerx,focal*XL(1)/XL(2)+centery);
          Eigen::Vector2d XR_proj(focal*XR(0)/XR(2)+centerx,focal*XR(1)/XR(2)+centery);

          xL.at<cv::Vec2f>(y,0) = cv::Vec2f(XL_proj(0),XL_proj(1));
          xR.at<cv::Vec2f>(y,0) = cv::Vec2f(XR_proj(0),XR_proj(1));
        }
        
        return true;
    }

    bool synthesize_column_linear( double focal, double centerx, double centery,
                                   double theta, double phi, double alpha,
                                   Keyframe &left, Keyframe &right,
                                   cv::Mat &left_image_float, cv::Mat &right_image_float,
                                   cv::Mat synth_column )
    {
        const int height = synth_column.rows;
        
        cv::Mat xL(height,1,CV_32FC2);
        cv::Mat xR(height,1,CV_32FC2);
        
        bool success = get_synthetic_column_maps( focal, centerx, centery, theta, phi, left, right, xL, xR );
        if ( !success ) return false;
        
        // remap left and right views
        cv::Mat I_L;
        cv::Mat I_R;
        cv::remap(left_image_float,I_L,xL,cv::noArray(),cv::INTER_LINEAR);
        cv::remap(right_image_float,I_R,xR,cv::noArray(),cv::INTER_LINEAR);
        
        cv::Mat result;
        cv::addWeighted(I_L,(1.0-alpha),I_R,alpha,0.,result);
        result.convertTo(synth_column, CV_8UC3);

        return true;
    }

    bool synthesize_column_flowbased( double focal, double centerx, double centery,
                          double theta, double phi, double alpha,
                          Keyframe &left, Keyframe &right,
                          cv::Mat &left_image_float, cv::Mat &right_image_float,
                          cv::Mat &forward_flow, cv::Mat &backward_flow,
                          cv::Mat &synth_column )
    {
        const int width = 1;
        const int height = synth_column.rows;
        
        cv::Mat xL(height,width,CV_32FC2);
        cv::Mat xR(height,width,CV_32FC2);
        
        bool success = get_synthetic_column_maps( focal, centerx, centery, theta, phi, left, right, xL, xR);
        if ( !success ) return false;
        
        /// compute plane-induced displacement
        cv::Mat v_LR;
        cv::Mat v_RL;
        cv::subtract( xR, xL, v_LR );
        cv::subtract( xL, xR, v_RL );

        // look up flows
        cv::Mat F_LR;
        cv::Mat F_RL;
        cv::remap(forward_flow, F_LR, xL, cv::noArray(), cv::INTER_LINEAR);
        cv::remap(backward_flow, F_RL, xR, cv::noArray(), cv::INTER_LINEAR);
        
        // compute corrected flows
        cv::Mat Fs_LR;
        cv::Mat Fs_RL;
        cv::subtract( v_LR, F_LR, Fs_LR );
        cv::subtract( v_RL, F_RL, Fs_RL );

        // compute maps for interpolation
        cv::Mat xs_L;
        cv::Mat xs_R;
        
        cv::addWeighted( xL, 1.0, Fs_LR, alpha, 0.0, xs_L );
        cv::addWeighted( xR, 1.0, Fs_RL, 1.0-alpha, 0.0, xs_R );
        
        // remap left and right views
        cv::Mat I_L;
        cv::Mat I_R;
        cv::remap(left_image_float,I_L,xs_L,cv::noArray(),cv::INTER_LINEAR);
        cv::remap(right_image_float,I_R,xs_R,cv::noArray(),cv::INTER_LINEAR);
        
        // alpha blend
        cv::Mat result;
        cv::addWeighted(I_L,(1.0-alpha),I_R,alpha,0.,result);
        result.convertTo(synth_column, CV_8UC3);
        
        return true;
    }

    void load_keyframes( const std::string & posespath, std::vector<Keyframe> &keyframes )
    {
        FILE *posesf = fopen(posespath.c_str(),"r");
        while ( true )
        {
            int index;
            double t[3];
            double r[3];
            
            int nread = fscanf(posesf,"%d %lf %lf %lf %lf %lf %lf\n",
                               &index,
                               t+0,t+1,t+2,
                               r+0,r+1,r+2);
            if ( nread != 7 ) break;
            
            Keyframe kf;
            kf.index = index;
            kf.t << t[0],t[1],t[2];
            kf.r << r[0],r[1],r[2];
            kf.R = so3exp(kf.r);
            
            keyframes.push_back(kf);
        }
        fclose(posesf);
    }

    void decompose_keyframe_rotations( std::vector<Keyframe> &keyframes )
    {
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &kf = keyframes[i];
            
            Eigen::Vector3d up(0,1,0);
            Eigen::Vector3d Rup = kf.R*up;
            std::cout << "Rup: " << Rup.transpose() << "\n";
            double angle = acos(up.dot(Rup));
            std::cout << "angle: " << angle*180/M_PI << "\n";
            
            if ( fabs(angle) > 0 )
            {
                Eigen::Vector3d v = up.cross(Rup);
                v /= v.norm();
                kf.Rxz = so3exp(v*angle);
            }
            else
            {
                kf.Rxz = Eigen::Matrix3d::Identity();
            }
            kf.Ry = kf.Rxz.transpose()*kf.R;
            
            Eigen::Vector3d ry = so3ln(kf.Ry);
            kf.theta = -ry(1);
        }
    }

    double compute_theta( const Eigen::Vector3d &c, const Eigen::Vector3d &up )
    {
      Eigen::Vector3d cproj = c - up * c.dot(up);
      Eigen::Vector3d x = Eigen::Vector3d(1,0,0);
      Eigen::Vector3d xproj = x - up * (x.dot(up));
      return atan2(xproj.cross(cproj).dot(up),xproj.dot(cproj))+M_PI;
    }

    void compute_thetas( std::vector<Keyframe> &keyframes )
    {
        Eigen::Vector3d up = Eigen::Vector3d(0,1,0);
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Eigen::Vector3d c = -keyframes[i].R.transpose() * keyframes[i].t;
            keyframes[i].theta = compute_theta( c, up );
        }
    }

    Eigen::Matrix3d get_rotation( const Eigen::Vector3d &from, const Eigen::Vector3d &to )
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        double angle = acos(from.dot(to));
        if ( fabs(angle) > 0 )
        {
            Eigen::Vector3d v = from.cross(to);
            v /= v.norm();
            R = so3exp(v*angle);
        }
        return R;
    }

    void estimate_plane( std::vector<Keyframe> &keyframes )
    {
        RayPairList centers(keyframes.size());
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &kf = keyframes[i];
            
            // get camera center
            Eigen::Vector3d c = -kf.R.transpose() * kf.t;
            Ray ray;
            ray.head(3) = c;
            centers[i] = std::make_pair( ray, ray );
        }
        
        std::vector<PlaneEstimator*> estimators(200);
        for ( int i = 0; i < estimators.size(); i++ ) estimators[i] = new PlaneEstimator;
        
        PreemptiveRANSAC<RayPairList, PlaneEstimator> plane_ransac;
        plane_ransac.inlier_threshold = 0.01;
        
        std::vector<bool> inliers;
        PlaneEstimator *best_estimator = NULL;
        int ninliers = plane_ransac.compute( centers.begin(), centers.end(), estimators, &best_estimator, inliers );
        std::cout << ninliers << "/" << centers.size() << " inliers\n";
        std::cout << best_estimator->normal << "\n";
        Eigen::Vector3d up = best_estimator->normal;
        if ( up(1) < 0 ) up = -up;
        
        // calculate rotation to correct up vector
        Eigen::Matrix3d correction = get_rotation( up, Eigen::Vector3d(0,1,0) );
        std::cout << "correction rotation:\n" << correction << "\n";
        std::cout << "corrected up vector:\n" << correction * up << "\n";
        
        // c = -R.' * t
        // newup = correction * up
        // newc = correction * c = - correction * R.' * t = - (R * correction.') .' * t
        
        // rotate all cameras to correct up vector
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &kf = keyframes[i];
            
            kf.R = kf.R * correction.transpose();
            kf.r = so3ln(kf.R);
        }

        // check if we should flip upside-down
        int nflip = 0;
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &kf = keyframes[i];
            if ( kf.R(1,1) < 0 ) nflip++;
        }
        
        if ( nflip > keyframes.size() / 2 )
        {
            std::cout << "FLIP\n";
            Eigen::Matrix3d Rflip = Eigen::Matrix3d::Identity();
            Rflip(1,1) = -1;
            Rflip(2,2) = -1;
            for ( int i = 0; i < keyframes.size(); i++ )
            {
                Keyframe &kf = keyframes[i];
                kf.R = kf.R * Rflip;
                kf.r = so3ln(kf.R);
            }
        }
    }

    cv::Mat convert_to_spherical(const Intrinsics &intrinsics, const cv::Mat &cylindrical )
    {
        const int height = cylindrical.cols/2;
        double vertfov = 2. * atan2(cylindrical.rows,2.*intrinsics.focal);
        
        // do vertical remap first
        // make lookup table for vertical remap
        std::vector<double> lut(height);
        
        double minphi = -M_PI/2;
        double maxphi = M_PI/2;
        double phistep = (maxphi-minphi)/(height-1);
        
        for ( int phinum = 0; phinum < height; phinum++ )
        {
            double phi = minphi + phinum*phistep;
            double H = tan(phi);
            lut[phinum] = intrinsics.focal*H+intrinsics.centery;
        }
        
        // now remap columns of cylindrical panorama
        cv::Mat spherical(height,cylindrical.cols,CV_8UC3);
        
        for ( int y = 0; y < height; y++ )
        {
            if ( std::isinf(lut[y]) || std::isnan(lut[y]) ) continue;
            if ( lut[y] < 0 || lut[y] >= cylindrical.rows ) continue;
            cv::Mat patch;
            cv::getRectSubPix(cylindrical,cv::Size(cylindrical.cols,1),cv::Point2f(cylindrical.cols/2.f,lut[y]),patch);
            patch.copyTo(spherical.row(y));
        }

        // resize horizontally if necessary
        cv::Mat spherical_resized;
        if ( spherical.cols != 2*height )
        {
            cv::resize( spherical, spherical_resized, cv::Size(2*height,height) );
        } else {
            spherical_resized = spherical;
        }
        
        return spherical_resized;
    }

    void make_stereo_panoramas( const Intrinsics &intrinsics, const std::string &videopath, const std::string &outputpath,
        const int panowidth, const bool is_loop )
    {
        const int start_theta = 0;
        const int end_theta = panowidth;
        
        std::string posespath = outputpath + "/poses.txt";
        
        std::vector<Keyframe> keyframes;
        
        std::cout << "loading keyframes from " << posespath << "...\n";
        load_keyframes( posespath, keyframes );
        std::cout << "loaded " << keyframes.size() << " keyframes\n";

        std::cout << "estimating plane...\n";
        Eigen::Vector3d up(0,1,0);
        estimate_plane( keyframes );

        std::cout << "decomposing rotations...\n";
        decompose_keyframe_rotations( keyframes );

        std::cout << "computing thetas...\n";
        compute_thetas( keyframes );

        std::cout << "keyframe poses: \n";
        for ( int i = 0; i < keyframes.size(); i++ ) std::cout << i << "\t" << keyframes[i].t.transpose() << "\t" << so3ln(keyframes[i].R).transpose() << "\n";

        std::cout << "thetas before: \n";
        for ( int i = 0; i < keyframes.size(); i++ ) std::cout << i << "\t" << keyframes[i].theta*180/M_PI << "\n";
        
        // re-order keyframes if necessary to make thetas increase
        std::cout << "re-ordering...\n";
        int npos = 0;
        int nneg = 0;
        for ( int i = 0; i < 10; i++ ) 
        {
            if ( keyframes[1].theta < keyframes[0].theta ) nneg++;
            else if ( keyframes[0].theta < keyframes[1].theta ) npos++;
        }
        bool reverse = ( nneg > npos );
        
        // remove end frames that overlap with beginning frames
        if ( is_loop )
        {
            if ( reverse )
            {
                while ( keyframes.back().theta < keyframes[0].theta )
                {
                    keyframes.pop_back();
                }
            } else {
                while ( keyframes.back().theta > keyframes[0].theta )
                {
                    keyframes.pop_back();
                }
            }
        }
        
        std::cout << "thetas after: \n";
        for ( int i = 0; i < keyframes.size(); i++ ) std::cout << i << "\t" << keyframes[i].theta*180/M_PI << "\n";
        
        std::cout << "loading images...\n";
        cv::VideoCapture cap(videopath);
        int video_index = -1;
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            while ( video_index < keyframes[i].index )
            {
                if ( !cap.read(keyframes[i].image) )
                {
                    std::cout << "could not read all keyframe images from " << videopath << "\n";
                    exit(1);
                }
                video_index++;
            }
        }

        const int width = keyframes[0].image.cols;
        const int height = keyframes[0].image.rows;
        
        const int nphi = 9;
        const double min_phi = (-(nphi-1)/2.)*M_PI/180.;
        const double max_phi = ((nphi-1)/2.)*M_PI/180.;
        std::vector<double> phirange(nphi);
        if ( nphi == 1 )
        {
          phirange[0] = 0;
        } else {
          for ( int i = 0; i < nphi; i++ ) phirange[i] = min_phi + i*(max_phi-min_phi)/(nphi-1);
        }
        
        const double min_theta = -M_PI;
        const double max_theta = M_PI;
        const int ntheta = panowidth;
        std::vector<double> thetarange(ntheta);
        const double theta_step = (max_theta-min_theta)/(ntheta-1);
        for ( int i = 0; i < ntheta; i++ ) thetarange[i] = min_theta + i*theta_step;
        
        std::cout << "interpolating frames...\n";
        
        int last_leftnum = -1;
        cv::Mat forward_flow;
        cv::Mat backward_flow;
        cv::Mat left_image;
        cv::Mat right_image;
        cv::Mat left_image_gray;
        cv::Mat right_image_gray;
        cv::Mat left_image_float;
        cv::Mat right_image_float;
        
        std::vector<cv::Mat> panoramas(nphi);
        for ( int phinum = 0; phinum < nphi; phinum++ )
        {
            panoramas[phinum] = cv::Mat(height,ntheta,CV_8UC3,cv::Scalar(0,0,0,0));
        }
        
        // iterate through each keyframe pair
        for ( int kfnum = 0; kfnum < keyframes.size(); kfnum++ )
        {
          if ( kfnum % 10 == 0 ) std::cout << kfnum+1 << " / " << keyframes.size() << "\n";
          if ( !is_loop && kfnum == keyframes.size()-1 ) break;

          int leftnum = kfnum;
          int rightnum = (kfnum+1)%keyframes.size();

          Keyframe &left = keyframes[leftnum];
          Keyframe &right = keyframes[rightnum];

          // load images
          left_image = left.image;
          cv::cvtColor( left_image, left_image_gray, cv::COLOR_BGR2GRAY );
          left_image.convertTo(left_image_float,CV_32FC3);
          right_image = right.image;
          right_image.convertTo(right_image_float,CV_32FC3);
          cv::cvtColor( right_image, right_image_gray, cv::COLOR_BGR2GRAY );

          // compute flow
          _compute_flow( left_image_gray, right_image_gray, forward_flow );
          _compute_flow( right_image_gray, left_image_gray, backward_flow );

          // get left and right camera centers
          Eigen::Vector3d C_L = -left.R.transpose() * left.t;
          Eigen::Vector3d C_R = -right.R.transpose() * right.t;

          bool found_one_theta = false;

          // find theta / phi combinations which fall between these keyframes
          for ( int thetanum = start_theta; thetanum < end_theta; thetanum++ )
          {
            double theta = thetarange[thetanum];
            Eigen::Vector3d synth_t(0,0,-synth_radius);
            Eigen::Vector3d synth_r(0,-theta,0);
            Eigen::Matrix3d synth_R = so3exp(synth_r);
            
            // get synthetic camera center
            Eigen::Vector3d C_D = -synth_R.transpose() * synth_t;

            // project left and right camera centers into synth camera
            Eigen::Vector3d r_L = C_L - C_D;
            Eigen::Vector3d r_R = C_R - C_D;
              
            // project rays to circle
            Eigen::Vector3d rs_L = project(r_L,up);
            Eigen::Vector3d rs_R = project(r_R,up);
            
            for ( int phinum = 0; phinum < phirange.size(); phinum++ ) 
            {
              double phi = phirange[phinum];

              // get synthetic ray in synth camera coordinate frame
              Eigen::Vector3d r_D(tan(phi),0,1);
              // get synthetic ray in world coordinate frame
              r_D = synth_R.transpose() * (r_D - synth_t);
              
              // project synthetic ray to circle
              Eigen::Vector3d rs_D = project(r_D,up);
              
              double angle_LD = signed_angle_between(rs_L,rs_D,up);
              double angle_RD = signed_angle_between(rs_R,rs_D,up);
              double angle_LR = signed_angle_between(rs_L,rs_R,up);

              // check that cameras lie on either side of synth ray
              if ( ! ( angle_LD * angle_RD < 0 ) ) continue;
              
              // check that cameras are within front hemisphere
              const double angle_thresh = M_PI/2;
              if ( fabs(angle_LD) >= angle_thresh ) continue;
              if ( fabs(angle_RD) >= angle_thresh ) continue;
            
              // calculate alpha
              double alpha = fabs(angle_LD)/fabs(angle_LR);
              //std::cout << alpha << "\n";
            
              // synthesize column
              cv::Mat synth_column(height,1,CV_8UC3);
            
              bool success = synthesize_column_flowbased( intrinsics.focal, intrinsics.centerx, intrinsics.centery, theta, phi, alpha, left, right, left_image_float, right_image_float, forward_flow, backward_flow, synth_column );
              if ( !success ) continue;
              found_one_theta = true;
              std::cout << theta << " " << phi << "\n";
            
              // store column in output panorama
              int colout = thetanum;
              int shift = round(phi/theta_step);
              colout = (colout+shift)%ntheta;
              if ( colout < 0 ) colout += ntheta;
                
              synth_column.copyTo(panoramas[phinum].col(colout));
            }
          }
          
        }
        
        std::vector<cv::Mat> spherical_panos(panoramas.size());
        for ( int i = 0; i < panoramas.size(); i++ )
        {
            spherical_panos[i] = convert_to_spherical(intrinsics, panoramas[i]);
        }
        for ( int phinum = 0; phinum < nphi; phinum++ )
        {
            cv::imwrite(outputpath + "/cylindrical" + std::to_string(phinum) + ".png", panoramas[phinum]);
            cv::imwrite(outputpath + "/spherical" + std::to_string(phinum) + ".png", spherical_panos[phinum]);
        }
        for ( int phinum = 0; phinum < nphi/2; phinum++ )
        {
            cv::Mat overunder;
            cv::vconcat(spherical_panos[nphi-phinum-1],spherical_panos[phinum],overunder);
            cv::imwrite(outputpath + "/overunder" + std::to_string(nphi-phinum-1) + std::to_string(phinum) + ".png", overunder);
            cv::imwrite(outputpath + "/overunder" + std::to_string(nphi-phinum-1) + std::to_string(phinum) + ".jpg", overunder);
        }
    }
}
