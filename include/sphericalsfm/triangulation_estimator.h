#pragma once

#include <Eigen/Core>
#include <vector>
#include <sphericalsfm/sfm_types.h>

namespace sphericalsfm {
    struct TriangulationObservation
    {
        Pose pose;
        Eigen::Vector2d x;
        double focal;
        TriangulationObservation( ) : x(Eigen::Vector2d::Zero()), focal(1) { }
        TriangulationObservation( const Pose &_pose, const Eigen::Vector2d &_x, const double &_focal ) : pose(_pose), x(_x), focal(_focal) { }
    };
    typedef std::vector<TriangulationObservation> TriangulationObservationList;
    class TriangulationEstimator
    {
        const TriangulationObservationList &observations;
    public:
        TriangulationEstimator(const TriangulationObservationList &_observations) : observations(_observations) { }

        inline int min_sample_size() const { return 2; }

        inline int non_minimal_sample_size() const { return 2; }

        inline int num_data() const { return observations.size(); }

        int MinimalSolver(const std::vector<int>& sample, std::vector<Point>* pts) const;

        // Returns 0 if no model could be estimated and 1 otherwise.
        int NonMinimalSolver(const std::vector<int>& sample, Point *pt) const;

        // Evaluates the pose on the i-th data point.
        double EvaluateModelOnPoint(const Point& pt, int i) const;

        // Linear least squares solver. Calls NonMinimalSolver.
        void LeastSquares(const std::vector<int>& sample, Point* pt) const;
    };
}

