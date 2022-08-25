
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace sphericalsfm {
    typedef Eigen::Vector3d Point;
    typedef Eigen::Matrix<double,6,1> Camera;

    // camera center should be removed
    typedef Eigen::Vector2d Observation;

    struct Pose
    {
        Eigen::Vector3d t;
        Eigen::Vector3d r;
        Eigen::Matrix4d P;
        
        Pose();
        Pose( const Eigen::Vector3d &_t, const Eigen::Vector3d &_r );
        Pose inverse() const;
        void postMultiply( const Pose &pose );
        Point apply( const Point &point ) const;
        Point applyInverse( const Point &point ) const;
        Eigen::Vector3d getCenter() const;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct Intrinsics
    {
        double focal, centerx, centery;
        Intrinsics( double _focal, double _centerx, double _centery ) :
        focal(_focal), centerx(_centerx), centery(_centery) { }
        Eigen::Matrix3d getK() const {
            Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
            K(0,0) = focal;
            K(1,1) = focal;
            K(0,2) = centerx;
            K(1,2) = centery;
            return K;
        }
        Eigen::Matrix3d getKinv() const {
            return getK().lu().inverse();
        }
    };

}

