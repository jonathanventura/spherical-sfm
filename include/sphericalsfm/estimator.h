#pragma once

#include <sphericalsfm/ray.h>

namespace sphericalsfm {
    struct Estimator
    {
        virtual int sampleSize() = 0;
        virtual int compute( RayPairList::iterator begin, RayPairList::iterator end ) = 0;
        virtual bool canRefine() { return false; }
        virtual void chooseSolution( int soln ) { }
        virtual double score( RayPairList::iterator it ) = 0;
        Estimator() { }
        virtual ~Estimator() { }
    };
}

