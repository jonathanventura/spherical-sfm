
#pragma once

#include <Eigen/Dense>
#include <sphericalsfm/ray.h>

namespace sphericalsfm {
    void make_spherical_essential_matrix( const Eigen::Matrix3d &R, bool inward, Eigen::Matrix3d &E );
    void decompose_spherical_essential_matrix( const Eigen::Matrix3d &E, bool inward, Eigen::Vector3d &r, Eigen::Vector3d &t );
    void decompose_spherical_essential_matrix( const Eigen::Matrix3d &E, bool inward,
                     RayPairList::iterator begin, RayPairList::iterator end, const std::vector<bool> &inliers,
                     Eigen::Vector3d &r, Eigen::Vector3d &t );
}
