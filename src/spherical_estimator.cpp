
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

#include <sphericalsfm/spherical_estimator.h>
#include <sphericalsfm/spherical_solvers.h>
#include <sphericalsfm/spherical_utils.h>
#include <sphericalsfm/so3.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

namespace sphericalsfm {

    struct SampsonError
    {
        template <typename T>
        bool operator()(const T* const ri_data,
                        const T* const ti_data,
                        const T* const rj_data,
                        const T* const tj_data,
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

            Eigen::Map< const Eigen::Matrix<T,3,1> > u(u_data);
            Eigen::Map< const Eigen::Matrix<T,3,1> > v(v_data);

            Eigen::Matrix<T,3,1> line = E * (u/u(2));
            T d = v.dot( line );
        
            residuals[0] = d / sqrt(line[0]*line[0] + line[1]*line[1]);
            
            return true;
        }
    };
    
    double SphericalEstimator::EvaluateModelOnPoint(const Eigen::Matrix3d& E, int i) const
    {
        if ( i > correspondences.size() ) { std::cout << "error: " << i << " / " << correspondences.size() << std::endl; }
        const RayPair &ray_pair(correspondences[i]);
        const Eigen::Vector3d &u = ray_pair.first;
        const Eigen::Vector3d &v = ray_pair.second;
        const Eigen::Vector3d line = E * (u/u(2));
        const double d = v.dot( line );
        
        return (d*d) / (line[0]*line[0] + line[1]*line[1]);
    }

    int SphericalEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es) const
    {
        if ( use_poly_solver) return spherical_solver_polynomial(correspondences, sample, Es);
        else return spherical_solver_action_matrix(correspondences, sample, Es);
    }
    
    int SphericalEstimator::NonMinimalSolver(const std::vector<int>& sample, Eigen::Matrix3d*E) const
    {
        std::vector<Eigen::Matrix3d> Es;
        spherical_solver_action_matrix(correspondences, sample, &Es);
        if ( Es.empty() ) return 0;
        double best_score = INFINITY;
        int best_ind = 0;
        for ( int i = 0; i < Es.size(); i++ )
        {
            double score = 0;
            for ( int j = 0; j < sample.size(); j++ )
            {
                score += EvaluateModelOnPoint(Es[i],sample[j]);
            }
            if ( score < best_score ) 
            {
                best_score = score;
                best_ind = i;
            }
        }
        *E = Es[best_ind];
        return 1;
    }

    void SphericalEstimator::LeastSquares(const std::vector<int>& sample, Eigen::Matrix3d* E) const
    {
        Eigen::Vector3d r0(0,0,0);
        Eigen::Vector3d t0(0,0,-1);
        if ( inward ) t0[2] = 1;
        Eigen::Vector3d r, t;
        decompose_spherical_essential_matrix( *E, inward, r, t );
        Eigen::Vector3d r1(r);
        Eigen::Vector3d t1(0,0,-1);
        if ( inward ) t1[2] = 1;
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
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SampsonError, 1, 3, 3, 3, 3, 3, 3>(error);
            problem.AddResidualBlock(cost_function, NULL,
                r0.data(), t0.data(),
                r1.data(), t1.data(),
                u[i].data(),
                v[i].data() );
            problem.SetParameterBlockConstant( u[i].data() );
            problem.SetParameterBlockConstant( v[i].data() );
        }
        problem.SetParameterBlockConstant( r0.data() );
        problem.SetParameterBlockConstant( t0.data() );

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.max_num_consecutive_invalid_steps = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 16;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        make_spherical_essential_matrix(so3exp(r1),inward,*E);
    }

    void SphericalEstimator::Decompose(const Eigen::Matrix3d &E, const std::vector<int> &inliers, Eigen::Matrix3d *R, Eigen::Vector3d *t) const
    {
        Eigen::Vector3d r;
        decompose_spherical_essential_matrix( E, inward, r, *t );
        *R = so3exp(r);
    }
}

