
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>
#include <sphericalsfm/ray.h>

using namespace sphericalsfm;

void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Eigen::Matrix3d* R, Eigen::Vector3d* t,
                             std::vector<Eigen::Vector3d>* points3D);

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2, Eigen::Vector3d* t);

class FivePointEstimator : public EssentialEstimator
{
protected:
    const RayPairList &correspondences;
public:
    FivePointEstimator(const RayPairList &_correspondences) : correspondences(_correspondences) { }

    inline int min_sample_size() const { return 5; }

    inline int non_minimal_sample_size() const { return 6; }

    inline int num_data() const { return correspondences.size(); }

    virtual int MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const = 0;

    // Returns 0 if no model could be estimated and 1 otherwise.
    virtual int NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const = 0;

    // Evaluates the pose on the i-th data point.
    double EvaluateModelOnPoint(const Eigen::Matrix3d& E, int i) const;

    // Linear least squares solver. Calls NonMinimalSolver.
    virtual void LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const = 0;

    void Decompose(const Eigen::Matrix3d &E, const std::vector<int> &inliers, Eigen::Matrix3d *R, Eigen::Vector3d *t) const;
};
