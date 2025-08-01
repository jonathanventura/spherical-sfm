
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>
#include <sphericalsfm/ray.h>

namespace sphericalsfm {
    class SphericalEstimator : public EssentialEstimator
    {
        const RayPairList &correspondences;
        const bool use_poly_solver;
        const bool inward;
    public:
        SphericalEstimator(const RayPairList &_correspondences, const bool _use_poly_solver, const bool _inward) : correspondences(_correspondences), use_poly_solver(_use_poly_solver), inward(_inward) { }

        inline int min_sample_size() const { return 3; }

        inline int non_minimal_sample_size() const { return 4; }

        inline int num_data() const { return correspondences.size(); }

        virtual int MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const Eigen::Matrix3d& E, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const;

        void Decompose(const Eigen::Matrix3d &E, const std::vector<int> &inliers, Eigen::Matrix3d *R, Eigen::Vector3d *t) const;
    };
}
