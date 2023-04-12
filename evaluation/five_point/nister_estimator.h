
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>
#include <sphericalsfm/ray.h>

#include "five_point_estimator.h"

using namespace sphericalsfm;

class NisterEstimator : public FivePointEstimator
{
public:
    NisterEstimator(const RayPairList &_correspondences) : FivePointEstimator(_correspondences) { }

    int MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    int NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const;

    // Linear least squares solver. Calls NonMinimalSolver.
    void LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const;
};
