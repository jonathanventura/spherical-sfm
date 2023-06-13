#pragma once

#include <Eigen/Core>
#include <sphericalsfm/rotation_averaging.h>

namespace sphericalsfm {

    double optimize_rotations_and_focal_length( 
        std::vector<Eigen::Matrix3d> &rotations, 
        const std::vector<RelativeRotation> &relative_rotations, 
        double &focal_length, 
        const double min_focal, const double max_focal,
        const double focal_guess, 
        bool inward );

}
