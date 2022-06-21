#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <sphericalsfm/similarity_estimator.h>
#include <sphericalsfm/so3.h>

using namespace sphericalsfm;

int main(int argc, char **argv)
{
    Eigen::Vector3d p0(0,0,0);
    Eigen::Vector3d p1(1,0,0);
    Eigen::Vector3d p2(0,1,0);
    double s = 10;
    Eigen::Matrix3d R = so3exp(Eigen::Vector3d(1,2,3));
    Eigen::Vector3d t = Eigen::Vector3d(3,4,5);
    RayPairList point_pairs;
    point_pairs.push_back( std::make_pair( p0, s*R*p0+t ) );
    point_pairs.push_back( std::make_pair( p1, s*R*p1+t ) );
    point_pairs.push_back( std::make_pair( p2, s*R*p2+t ) );
    SimilarityEstimator estimator( point_pairs );
    Similarity sim;
    std::vector<int> sample;
    sample.push_back(0);
    sample.push_back(1);
    sample.push_back(2);
    estimator.NonMinimalSolver(sample, &sim);
    estimator.LeastSquares(sample, &sim);
    std::cout << "after least squares: " << estimator.EvaluateModelOnPoint(sim, 0) << "\n";
    std::cout << "after least squares: " << estimator.EvaluateModelOnPoint(sim, 1) << "\n";
    std::cout << "after least squares: " << estimator.EvaluateModelOnPoint(sim, 2) << "\n";
}
