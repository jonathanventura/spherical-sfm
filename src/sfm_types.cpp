
#include <Eigen/Geometry>

#include <sphericalsfm/sfm_types.h>
#include <sphericalsfm/so3.h>

namespace sphericalsfm {
    Pose::Pose()
    : t( Eigen::Vector3d::Zero() ), r( Eigen::Vector3d::Zero() ), P( Eigen::Matrix4d::Identity() )
    {
        
    }

    Pose::Pose( const Eigen::Vector3d &_t, const Eigen::Vector3d &_r )
    : t( _t ), r( _r ), P( Eigen::Matrix4d::Identity() )
    {
        P.block<3,3>(0,0) = so3exp(r);
        P.block<3,1>(0,3) = t;
    }

    Pose Pose::inverse() const
    {
        Pose poseinv;
        poseinv.P.block<3,3>(0,0) = P.block<3,3>(0,0).transpose();
        poseinv.t = -P.block<3,3>(0,0).transpose()*t;
        poseinv.r = -r;
        poseinv.P.block<3,1>(0,3) = poseinv.t;
        return poseinv;
    }

    void Pose::postMultiply( const Pose &pose )
    {
        P = P * pose.P;
        t = P.block<3,1>(0,3);
        r = so3ln( P.block<3,3>(0,0) );
    }

    Point Pose::apply( const Point &point ) const
    {
        return P.block<3,3>(0,0) * point + t;
    }

    Point Pose::applyInverse( const Point &point ) const
    {
        return P.block<3,3>(0,0).transpose() * ( point - t );
    }

    Eigen::Vector3d Pose::getCenter() const
    {
        return P.block<3,3>(0,0).transpose() * (-t);
    }
}
