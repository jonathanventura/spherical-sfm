
#pragma once

#include <Eigen/Core>
#include <sphericalsfm/estimator.h>

namespace sphericalsfm {
    typedef Eigen::Vector3d Point;
    typedef std::vector<Point> PointList;
    struct Plane
    {
        Eigen::Vector3d normal;
        double d;
    };
    struct PlaneEstimator : public Estimator<Plane>
    {
        const PointList &points;
    public:
        PlaneEstimator(const PointList &_points ) : points(_points) { }

        int min_sample_size() const { return 3; }

        inline int non_minimal_sample_size() const { return 4; }

        inline int num_data() const { return points.size(); }

        int MinimalSolver(const std::vector<int>& sample, std::vector<Plane>* planes) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, Plane* plane) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const Plane& plane, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, Plane* plane) const;
    };
}

