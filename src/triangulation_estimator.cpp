
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>

#include <sphericalsfm/triangulation_estimator.h>
#include <sphericalsfm/sfm_types.h>
#include <sphericalsfm/so3.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

namespace sphericalsfm {
    
    struct TriangulationError
    {
        template <typename T>
        bool operator()(const T* const r_data,
                        const T* const t_data,
                        const T* const X_data,
                        const T* const obs_data,
                        const T* const focal_data,
                        T* residuals) const
        {
            // proj = R p + t
            T PX_data[3];
            ceres::AngleAxisRotatePoint(r_data,X_data,PX_data);
            PX_data[0] += t_data[0];
            PX_data[1] += t_data[1];
            PX_data[2] += t_data[2];
            
            T proj_data[2];
            proj_data[0] = PX_data[0] / PX_data[2];
            proj_data[1] = PX_data[1] / PX_data[2];

            residuals[0] = (*focal_data) * proj_data[0] - obs_data[0];
            residuals[1] = (*focal_data) * proj_data[1] - obs_data[1];
            
            return true;
        }
    };
    
    double TriangulationEstimator::EvaluateModelOnPoint(const Point& pt, int i) const
    {
        const TriangulationObservation &obs(observations[i]);
        const Eigen::Vector3d PX = obs.pose.apply(pt);
        if ( PX(2) < 0 ) return std::numeric_limits<double>::max();
        const Eigen::Vector2d proj = PX.head(2)/PX(2);
        const Eigen::Vector2d residuals = obs.focal * proj - obs.x;
        return residuals.squaredNorm();
    }

    int TriangulationEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Point>* pts) const
    {
        Point pt;
        if ( !NonMinimalSolver(sample,&pt) ) return 0;
        pts->clear();
        pts->push_back(pt);
        return 1;
    }
    
    int TriangulationEstimator::NonMinimalSolver(const std::vector<int>& sample, Point*pt) const
    {
        const int N = sample.size();

        Eigen::MatrixXd A( N*2, 4 );
         
        for ( int n = 0; n < N; n++ )
        {
            const TriangulationObservation &obs = observations[sample[n]];

            const Eigen::Vector2d point = obs.x/obs.focal;
            
            A.row(2*n+0) = obs.pose.P.row(2) * point(0) - obs.pose.P.row(0);
            A.row(2*n+1) = obs.pose.P.row(2) * point(1) - obs.pose.P.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A,Eigen::ComputeFullV);
        Eigen::Vector4d Xh = svdA.matrixV().col(3);
        *pt = Xh.head(3)/Xh(3);
    
        return 1;
    }

    void TriangulationEstimator::LeastSquares(const std::vector<int>& sample, Point* pt) const
    {
        std::vector<Eigen::Vector3d> r;
        std::vector<Eigen::Vector3d> t;
        std::vector<Eigen::Vector2d> obs;
        std::vector<double> focal;
        
        ceres::Problem problem;

        for ( int ind : sample ) 
        {
            r.push_back(observations[ind].pose.r);
            t.push_back(observations[ind].pose.t);
            obs.push_back(observations[ind].x);
            focal.push_back(observations[ind].focal);
        }

        for ( int i = 0; i < sample.size(); i++ )
        {
            TriangulationError *error = new TriangulationError;
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<TriangulationError, 2, 3, 3, 3, 2, 1>(error);
            problem.AddResidualBlock(cost_function, NULL,
                r[i].data(), t[i].data(),
                pt->data(), obs[i].data(), &(focal[i]) );
            problem.SetParameterBlockConstant( r[i].data() );
            problem.SetParameterBlockConstant( t[i].data() );
            problem.SetParameterBlockConstant( obs[i].data() );
            problem.SetParameterBlockConstant( &(focal[i]) );
        }

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.max_num_consecutive_invalid_steps = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
}

