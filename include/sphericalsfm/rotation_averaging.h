#pragma once

#include <Eigen/Core>

namespace sphericalsfm {

    struct RelativeRotation
    {
        int index0, index1;
        Eigen::Matrix3d R; // R1 * R0^T
        RelativeRotation( const int _index0,  const int _index1, const Eigen::Matrix3d &_R ) : index0(_index0), index1(_index1), R(_R) { }
    };

    void optimize_rotations( std::vector<Eigen::Matrix3d> &rotations, const std::vector<RelativeRotation> &relative_rotations );

}
