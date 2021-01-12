#pragma once

#include <Eigen/Core>
#include <sphericalsfm/ray.h>

namespace sphericalsfm {
    struct Similarity
    {
        // q = s R p + t
        double s;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        Similarity( ) : s(1), R(Eigen::Matrix3d::Identity()), t(0,0,0) { }
        Similarity( const double &_s, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_t ) : s(_s), R(_R), t(_t) { }
    };
    class SimilarityEstimator
    {
        const RayPairList &correspondences;
    public:
        SimilarityEstimator(const RayPairList &_correspondences) : correspondences(_correspondences) { }

        inline int min_sample_size() const { return 3; }

        inline int non_minimal_sample_size() const { return 3; }

        inline int num_data() const { return correspondences.size(); }

        int MinimalSolver(const std::vector<int>& sample, std::vector<Similarity>* sims) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, Similarity *sim) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const Similarity& sim, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, Similarity* sim) const;
    };
}
