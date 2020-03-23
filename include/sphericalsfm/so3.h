
#pragma once

#include <Eigen/Core>

namespace sphericalsfm
{
    Eigen::Matrix3d skew3( const Eigen::Vector3d &v );
    Eigen::Matrix3d so3exp( const Eigen::Vector3d &r );
    Eigen::Vector3d so3ln( const Eigen::Matrix3d &R );
}

