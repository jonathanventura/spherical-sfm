#pragma once

#include <Eigen/Core>
#include <vector>
#include <algorithm>

namespace sphericalsfm {
    typedef Eigen::Matrix<double,3,1> Ray;
    typedef std::pair<Ray,Ray> RayPair;
    typedef std::vector<RayPair> RayPairList;
}

