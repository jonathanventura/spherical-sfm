
#include "stewenius_estimator.h"
#include "solver_stewenius.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

int SteweniusEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const
{
    const int N = sample.size();
    if ( N < 5 )
    {
        std::cout << "bad sample size: " << N << "\n";
        return 0;
    }

    Eigen::MatrixXd A(N,9);
    for ( int i = 0; i < N; i++ )
    {
        const Eigen::Vector3d u = correspondences[sample[i]].first;
        const Eigen::Vector3d v = correspondences[sample[i]].second;
        A.row(i) << u(0)*v(0), u(0)*v(1), u(0)*v(2), u(1)*v(0), u(1)*v(1), u(1)*v(2), u(2)*v(0), u(2)*v(1), u(2)*v(2);
    }
    
    Eigen::MatrixXd Q = A.jacobiSvd(Eigen::ComputeFullV).matrixV();
    // QR(A.') --> last columns of Q are nullspace
    //Eigen::MatrixXd Q = A.transpose().colPivHouseholderQr().householderQ();
    Eigen::Matrix<double,9,4> B = Q.block(0,5,9,4);
    
    std::vector<Eigen::MatrixXcd> w;
    solver_stewenius(B,&w);
    
    Es->clear();
    for ( int i = 0; i < w.size(); i++ )
    {
        const double x = w[i](0).real();
        const double y = w[i](1).real();
        const double z = w[i](2).real();
        Eigen::Vector4d bsoln( x, y, z, 1 );
        Eigen::Matrix<double,9,1> psoln( B*bsoln );
        
        Eigen::Matrix3d Esoln;
        Esoln <<
        psoln(0), psoln(3), psoln(6),
        psoln(1), psoln(4), psoln(7),
        psoln(2), psoln(5), psoln(8);
        
        Esoln /= Esoln.norm();
        
        Es->push_back(Esoln);
    }
    
    return Es->size();
}
    
int SteweniusEstimator::NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const
{
    return 0;
}

void SteweniusEstimator::LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const
{

}

