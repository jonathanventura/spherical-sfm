
#include <sphericalsfm/so3.h>
#include <sphericalsfm/pose_estimator.h>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>

namespace sphericalsfm
{
    int PoseEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Pose>* poses) const
    {
        Pose pose;
        if ( !NonMinimalSolver(sample,&pose) ) return 0;
        poses->clear();
        poses->push_back(pose);
        return 1;
    }

    // Returns 0 if no model could be estimated and 1 otherwise.
    int PoseEstimator::NonMinimalSolver(const std::vector<int>& sample, Pose*  pose) const
    {
        if ( sample.size() < 4 ) return 0;
        //if ( sample.size() > 4 ) return 0;
        
        std::vector<cv::Point3f> objectPoints(sample.size());
        std::vector<cv::Point2f> imagePoints(sample.size());
        
        for ( int i = 0; i < sample.size(); i++ )
        {
            const RayPair &corr = correspondences[sample[i]];
            objectPoints[i] = cv::Point3f(corr.first(0),corr.first(1),corr.first(2));
            imagePoints[i] = cv::Point2f(corr.second(0)/corr.second(2),corr.second(1)/corr.second(2));
        }

        cv::Mat cameraMatrix(3,3,CV_64F);
        cv::setIdentity(cameraMatrix);
        
        cv::Mat rvec, tvec;
        cv::solvePnP(objectPoints, imagePoints, cameraMatrix, cv::noArray(), rvec, tvec, false, (sample.size()==4) ? cv::SOLVEPNP_P3P : cv::SOLVEPNP_EPNP);
        
        if ( !rvec.data ) return 0;

        Eigen::Vector3d r, t;
        cv::cv2eigen(rvec,r);
        cv::cv2eigen(tvec,t);
        *pose = Pose(t,r);
        
        return 1;
    }

    // Evaluates the pose on the i-th data point.
    double PoseEstimator::EvaluateModelOnPoint(const Pose &pose, int i) const
    {
        const Eigen::Vector3d X = correspondences[i].first;
        const Eigen::Vector2d x(correspondences[i].second.head(2)/correspondences[i].second(2));
        const Eigen::Vector3d PX = pose.apply(X);
        if ( PX(2) < 0 ) return std::numeric_limits<double>::max();
        const Eigen::Vector2d proj = PX.head(2)/PX(2);
        const Eigen::Vector2d residuals = proj - x;
        return residuals.squaredNorm();
    }

    // Linear least squares solver. Calls NonMinimalSolver.
    void PoseEstimator::LeastSquares(const std::vector<int>& sample, Pose* pose) const
    {
        //return;
        if ( sample.size() < 6 ) return;
        
        std::vector<cv::Point3f> objectPoints(sample.size());
        std::vector<cv::Point2f> imagePoints(sample.size());
        
        for ( int i = 0; i < sample.size(); i++ )
        {
            const RayPair &corr = correspondences[sample[i]];
            objectPoints[i] = cv::Point3f(corr.first(0),corr.first(1),corr.first(2));
            imagePoints[i] = cv::Point2f(corr.second(0)/corr.second(2),corr.second(1)/corr.second(2));
        }

        cv::Mat cameraMatrix(3,3,CV_64F);
        cv::setIdentity(cameraMatrix);
        
        cv::Mat rvec, tvec;
        cv::eigen2cv(pose->r,rvec);
        cv::eigen2cv(pose->t,tvec);
        cv::solvePnP(objectPoints, imagePoints, cameraMatrix, cv::noArray(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        
        if ( !rvec.data ) return;
        
        Eigen::Vector3d r, t;
        cv::cv2eigen(rvec,r);
        cv::cv2eigen(tvec,t);
        *pose = Pose(t,r);
        
        return;
    }

}
