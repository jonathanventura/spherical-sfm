
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>

#include <sphericalsfm/plane_estimator.h>
#include <sphericalsfm/so3.h>

namespace sphericalsfm {
    double PlaneEstimator::EvaluateModelOnPoint(const Plane& plane, int i) const
    {
        const Eigen::Vector3d &x = points[i];
        const double proj = plane.normal.dot( x ) + plane.d;
        return proj*proj;
    }

    int PlaneEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Plane>* planes) const
    {
        const int N = sample.size();
        
        Eigen::MatrixXd A(N,4);
        
        for ( int i = 0; i < N; i++ )
        {
            const Eigen::Vector3d x = points[sample[i]];
            
            A(i,0) = x(0);
            A(i,1) = x(1);
            A(i,2) = x(2);
            A(i,3) = 1;
        }
        
        Eigen::Matrix4d B = A.jacobiSvd(Eigen::ComputeFullV).matrixV();
        Eigen::Vector4d soln = B.col(3);
        soln = soln/soln.head(3).norm();
        
        Plane plane;
        plane.normal = soln.head(3);
        plane.d = soln(3);
        
        planes->clear();
        planes->push_back(plane);

        return 1;
    }

    // Returns 0 if no model could be estimated and 1 otherwise.
    int PlaneEstimator::NonMinimalSolver(const std::vector<int>& sample, Plane* plane) const
    {
        std::vector<Plane> planes;
        MinimalSolver(sample,&planes);
        *plane = planes[0];
        return 1;
    }

    // Linear least squares solver. Calls NonMinimalSolver.
    void PlaneEstimator::LeastSquares(const std::vector<int>& sample, Plane* plane) const
    {
        NonMinimalSolver(sample,plane);
    }
}

