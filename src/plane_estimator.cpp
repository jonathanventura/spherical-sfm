
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>

#include <sphericalsfm/plane_estimator.h>
#include <sphericalsfm/so3.h>

namespace sphericalsfm {
    int PlaneEstimator::sampleSize()
    {
        return 3;
    }

    double PlaneEstimator::score( RayPairList::iterator it )
    {
        const Eigen::Vector3d &x = it->first.head(3);
        const double proj = normal.dot( x ) + d;
        return proj*proj;
    }

    bool PlaneEstimator::canRefine()
    {
        return true;
    }

    int PlaneEstimator::compute( RayPairList::iterator begin, RayPairList::iterator end )
    {
        int N = std::distance(begin,end);
        
        Eigen::MatrixXd A(N,4);
        
        int i = 0;
        for ( RayPairList::iterator it = begin; it != end; it++,i++ )
        {
            const Eigen::Vector3d x = it->first.head(3);
            
            A(i,0) = x(0);
            A(i,1) = x(1);
            A(i,2) = x(2);
            A(i,3) = 1;
        }
        
        Eigen::Matrix4d B = A.jacobiSvd(Eigen::ComputeFullV).matrixV();
        Eigen::Vector4d soln = B.col(3);
        soln = soln/soln.head(3).norm();
        normal = soln.head(3);
        d = soln(3);
        
        return 1;
    }
}

