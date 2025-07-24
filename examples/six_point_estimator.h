#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>
#include <sphericalsfm/ray.h>
#include <sphericalsfm/sfm_types.h>

namespace sphericalsfmtools {
    struct SixPointSolution
    {
        sphericalsfm::Pose pose;
        double focal;
    };

    class SixPointEstimator : public sphericalsfm::Estimator<SixPointSolution>
    {
        const sphericalsfm::RayPairList &correspondences;
    public:
        SixPointEstimator(const sphericalsfm::RayPairList &_correspondences) : correspondences(_correspondences) { }

        inline int min_sample_size() const { return 6; }

        inline int non_minimal_sample_size() const { return 7; }

        inline int num_data() const { return correspondences.size(); }

        virtual int MinimalSolver(const std::vector<int>& sample, std::vector<SixPointSolution>* solns) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, SixPointSolution*soln) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const SixPointSolution& soln, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, SixPointSolution* soln) const;
    };
}
