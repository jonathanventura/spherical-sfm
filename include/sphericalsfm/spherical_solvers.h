
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/ray.h>

namespace sphericalsfm {
    int spherical_solver_action_matrix(const RayPairList &correspondences, const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es);
    int spherical_solver_polynomial(const RayPairList &correspondences, const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es);
}
