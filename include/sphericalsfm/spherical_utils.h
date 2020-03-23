
#pragma once

#include <Eigen/Dense>

namespace sphericalsfm {
    void make_spherical_essential_matrix( const Eigen::Matrix3d &R, bool inward, Eigen::Matrix3d &E );
    void decompose_spherical_essential_matrix( const Eigen::Matrix3d &E, bool inward, Eigen::Vector3d &r, Eigen::Vector3d &t );
}
