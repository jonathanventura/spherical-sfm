
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>

namespace sphericalsfm {
    struct SphericalEstimator : public Estimator
    {
        Eigen::Matrix3d Esolns[4];
        
        Eigen::Matrix3d E;
        
        int sampleSize();
        double score( RayPairList::iterator it );
        void chooseSolution( int soln );
        int compute( RayPairList::iterator begin, RayPairList::iterator end );
        bool canRefine();
        
        // R,t decomposition with cheirality testing
        void decomposeE( bool inward,
                        RayPairList::iterator begin, RayPairList::iterator end, const std::vector<bool> &inliers,
                        Eigen::Vector3d &r, Eigen::Vector3d &t );
    };
}
