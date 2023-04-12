
#include "nister_estimator.h"

#include <PoseLib/solvers/relpose_5pt.h>
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>

int NisterEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const
{
    std::vector<Eigen::Vector3d> x1;
    std::vector<Eigen::Vector3d> x2;
    for ( int i = 0; i < 5; i++ ) {
        x1.push_back(correspondences[sample[i]].first.normalized());
        x2.push_back(correspondences[sample[i]].second.normalized());
    }
    return poselib::relpose_5pt(x1,x2,Es);
}
    
int NisterEstimator::NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const
{
    return 0;
}

void NisterEstimator::LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const
{

}

