#pragma once

#include <vector>

namespace sphericalsfm {
    template<typename SolutionType>
    class Estimator
    {
    public:
        virtual inline int min_sample_size() const = 0;

        virtual inline int non_minimal_sample_size() const = 0;

        virtual inline int num_data() const = 0;

        virtual int MinimalSolver(const std::vector<int>& sample, std::vector<SolutionType>* Es) const = 0;

        virtual int NonMinimalSolver(const std::vector<int>& sample, SolutionType* E) const = 0;

        virtual double EvaluateModelOnPoint(const SolutionType& E, int i) const = 0;

        virtual void LeastSquares(const std::vector<int>& sample, SolutionType* E) const = 0;
    };

    class EssentialEstimator : public Estimator<Eigen::Matrix3d>
    {
    public:
        virtual void Decompose(const Eigen::Matrix3d &E, Eigen::Matrix3d *R, Eigen::Vector3d *t) const = 0;
    };
}

