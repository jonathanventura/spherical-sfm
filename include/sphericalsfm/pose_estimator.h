
#pragma once

#include <sphericalsfm/ray.h>
#include <sphericalsfm/sfm_types.h>

namespace sphericalsfm
{
    class PoseEstimator
    {
        const RayPairList &correspondences;
    public:
        PoseEstimator(const RayPairList &_correspondences)
        : correspondences(_correspondences) { }

        inline int min_sample_size() const { return 4; }

        inline int non_minimal_sample_size() const { return 4; }

        inline int num_data() const { return correspondences.size(); }

        int MinimalSolver(const std::vector<int>& sample, std::vector<Pose>* poses) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, Pose* pose) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const Pose& pose, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, Pose* pose) const;
    };

}
