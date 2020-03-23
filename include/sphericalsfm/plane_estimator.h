
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>

namespace sphericalsfm {
    struct PlaneEstimator : public Estimator
    {
        Eigen::Vector3d normal;
        double d;
        
        int sampleSize();
        double score( RayPairList::iterator it );
        int compute( RayPairList::iterator begin, RayPairList::iterator end );
        bool canRefine();
    };
}

