
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

#include <PoseLib/solvers/relpose_6pt_focal.h>

#include "six_point_estimator.h"
#include <sphericalsfm/so3.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

using namespace sphericalsfm;

namespace sphericalsfmtools {

    struct SampsonError
    {
        template <typename T>
        bool operator()(const T* const ri_data,
                        const T* const ti_data,
                        const T* const rj_data,
                        const T* const tj_data,
                        const T* const focal_data,
                        const T* const u_data,
                        const T* const v_data,
                        T* residuals) const
        {
            T Ri_data[9];
            ceres::AngleAxisToRotationMatrix(ri_data,Ri_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Ri(Ri_data);
            Eigen::Map< const Eigen::Matrix<T,3,1> > ti(ti_data);

            T Rj_data[9];
            ceres::AngleAxisToRotationMatrix(rj_data,Rj_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rj(Rj_data);
            Eigen::Map< const Eigen::Matrix<T,3,1> > tj(tj_data);
            
            Eigen::Matrix<T,3,3> R = Rj*Ri.transpose();
            Eigen::Matrix<T,3,1> t = Rj*(-Ri.transpose()*ti) + tj;
            
            Eigen::Matrix<T,3,3> s;
            s <<
            T(0), -t[2], t[1],
            t[2], T(0), -t[0],
            -t[1], t[0], T(0);
            Eigen::Matrix<T,3,3> E = s * R;

            Eigen::Matrix<T,3,3> Kinv;
            Kinv <<
            T(1), T(0), T(0),
            T(0), T(1), T(0),
            T(0), T(0), (*focal_data);

            Eigen::Map< const Eigen::Matrix<T,3,1> > u(u_data);
            Eigen::Map< const Eigen::Matrix<T,3,1> > v(v_data);

            Eigen::Matrix<T,3,3> F = Kinv * E * Kinv;

            Eigen::Matrix<T,3,1> Fu = F * u;
            Eigen::Matrix<T,3,1> Ftv = F.transpose() * v;

            T d = v.dot( Fu );
            residuals[0] = (d*d) / ( Fu.head(2).squaredNorm() + Ftv.head(2).squaredNorm() );
            
            return true;
        }
    };

    double SixPointEstimator::EvaluateModelOnPoint(const SixPointSolution& soln, int i) const
    {
        if ( i > correspondences.size() ) { std::cout << "error: " << i << " / " << correspondences.size() << std::endl; }
        const RayPair &ray_pair(correspondences[i]);
        const Eigen::Vector3d &u = ray_pair.first;
        const Eigen::Vector3d &v = ray_pair.second;

        const Eigen::Matrix3d E = skew3(soln.pose.t) * so3exp(soln.pose.r);
        
        const Eigen::Vector3d Eu = E * u;
        const Eigen::Vector3d Etv = E.transpose() * v;
        
        const double d = v.dot( Eu );
        return (d*d) / ( Eu.head(2).squaredNorm() + Etv.head(2).squaredNorm() );
    }

    int SixPointEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<SixPointSolution>* solns) const
    {
        std::vector<Eigen::Vector3d> x1(6);
        std::vector<Eigen::Vector3d> x2(6);
        for ( int i = 0; i < 6; i++ )
        {
            x1[i] = correspondences[sample[i]].first;
            x2[i] = correspondences[sample[i]].second;
        }
        poselib::ImagePairVector out_image_pairs;
        int nsols = relpose_6pt_shared_focal(x1, x2, &out_image_pairs);
        
        solns->clear();
        for ( int i = 0; i < nsols; i++ )
        {
            SixPointSolution soln;

            soln.pose = Pose( out_image_pairs[i].pose.t, so3ln(out_image_pairs[i].pose.R()) );
            soln.focal = out_image_pairs[i].camera1.params[0];

            solns->push_back(soln);
        }

        return nsols;
    }
    
    int SixPointEstimator::NonMinimalSolver(const std::vector<int>& sample, SixPointSolution*soln) const
    {
        std::vector<SixPointSolution> solns;
        int nsols = MinimalSolver( sample, &solns );
        if ( nsols == 0 ) return 0;
        double best_score = INFINITY;
        int best_ind = 0;
        for ( int i = 0; i < nsols; i++ )
        {
            double score = 0;
            for ( int j = 0; j < sample.size(); j++ )
            {
                score += EvaluateModelOnPoint(solns[i],sample[j]);
            }
            if ( score < best_score ) 
            {
                best_score = score;
                best_ind = i;
            }
        }
        *soln = solns[best_ind];
        return 1;
    }

    void SixPointEstimator::LeastSquares(const std::vector<int>& sample, SixPointSolution* soln) const
    {
        Eigen::Vector3d r0(0,0,0);
        Eigen::Vector3d t0(0,0,0);
        Eigen::Vector3d r1 = soln->pose.r;
        Eigen::Vector3d t1 = soln->pose.t;
        double focal = soln->focal;
        std::vector<Eigen::Vector3d> u;
        std::vector<Eigen::Vector3d> v;
        
        ceres::Problem problem;

        for ( int ind : sample ) 
        {
            u.push_back(correspondences[ind].first);
            v.push_back(correspondences[ind].second);
        }

        for ( int i = 0; i < sample.size(); i++ )
        {
            SampsonError *error = new SampsonError;
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SampsonError, 1, 3, 3, 3, 3, 1, 3, 3>(error);
            problem.AddResidualBlock(cost_function, NULL,
                r0.data(), t0.data(),
                r1.data(), t1.data(),
                &focal,
                u[i].data(),
                v[i].data() );
            problem.SetParameterBlockConstant( u[i].data() );
            problem.SetParameterBlockConstant( v[i].data() );
        }
        problem.SetParameterBlockConstant( r0.data() );
        problem.SetParameterBlockConstant( t0.data() );
        problem.SetManifold(t1.data(), new ceres::SphereManifold<3>());

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.max_num_consecutive_invalid_steps = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 16;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        soln->pose = Pose( t1, r1 );
        soln->focal = focal;
    }
}

