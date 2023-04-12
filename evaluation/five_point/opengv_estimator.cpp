
#include "opengv_estimator.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>

using namespace opengv;

int OpenGVEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const
{
    const int N = sample.size();
    if ( N < 5 )
    {
        std::cout << "bad sample size: " << N << "\n";
        return 0;
    }

    bearingVectors_t bearingVectors1;
    bearingVectors_t bearingVectors2;

    for ( int i = 0; i < N; i++ )
    {
        bearingVectors1.push_back(correspondences[sample[i]].first);
        bearingVectors2.push_back(correspondences[sample[i]].second);
    }

    // create the central adapter
    relative_pose::CentralRelativeAdapter adapter( bearingVectors1, bearingVectors2 );

    // Stewenius' 5-point algorithm
    
    complexEssentials_t fivept_stewenius_essentials = relative_pose::fivept_stewenius( adapter );
    
    Es->clear();
    for ( int i = 0 ; i < fivept_stewenius_essentials.size(); i++ )
    {
        Es->push_back(fivept_stewenius_essentials[i].real().transpose());
    }
    
    return Es->size();
}
    
int OpenGVEstimator::NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const
{
    return 0;
}

void OpenGVEstimator::LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const
{

}

