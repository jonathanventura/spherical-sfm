
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>

#include <sphericalsfm/similarity_estimator.h>
#include <sphericalsfm/spherical_utils.h>
#include <sphericalsfm/so3.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

namespace sphericalsfm {
    
    Similarity umeyama( const Eigen::MatrixXd &p, const Eigen::MatrixXd &q )
    {
        int N = p.cols();
        Eigen::Vector3d meanp = Eigen::Vector3d::Zero();
        Eigen::Vector3d meanq = Eigen::Vector3d::Zero();
        for ( int i = 0; i < N; i++ )
        {
            meanp += p.col(i);
            meanq += q.col(i);
        }
        meanp /= N;
        meanq /= N;
        
        double sigmap = 0;
        double sigmaq = 0;
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for ( int i = 0; i < N; i++ )
        {
            const Eigen::Vector3d myp = p.col(i) - meanp;
            const Eigen::Vector3d myq = q.col(i) - meanq;
            sigmap += myp.squaredNorm();
            sigmaq += myq.squaredNorm();
            cov += myq*myp.transpose();
        }
        sigmap /= N;
        sigmaq /= N;
        cov /= N;
        
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov,Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Vector3d D = svd.singularValues();
        const Eigen::Matrix3d V = svd.matrixU();
        const double covdet = cov.determinant();
        const double Udet = U.determinant();
        const double Vdet = V.determinant();
        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
        if ( covdet < 0 || (covdet == 0 && Udet*Vdet < 0 ) )
        {
            S(2,2) = -1;
        }
        
        const Eigen::Matrix3d R = U * S * V.transpose();
        const double s = 1./sigmap * (D.asDiagonal()*S).trace();
        const Eigen::Vector3d t = meanq - s*R*meanp;
        return Similarity(s,R,t);
    }
     
    struct SimilarityError
    {
        template <typename T>
        bool operator()(const T* const s_data,
                        const T* const r_data,
                        const T* const t_data,
                        const T* const p_data,
                        const T* const q_data,
                        T* residuals) const
        {
            // q = s R p + t
            T pp_data[3];
            ceres::AngleAxisRotatePoint(r_data,p_data,pp_data);
            pp_data[0] = (*s_data) * pp_data[0] + t_data[0];
            pp_data[1] = (*s_data) * pp_data[1] + t_data[1];
            pp_data[2] = (*s_data) * pp_data[2] + t_data[2];

            residuals[0] = q_data[0] - pp_data[0];
            residuals[1] = q_data[1] - pp_data[1];
            residuals[2] = q_data[2] - pp_data[2];
            
            return true;
        }
    };
    
    double SimilarityEstimator::EvaluateModelOnPoint(const Similarity& sim, int i) const
    {
        const RayPair &ray_pair(correspondences[i]);
        const Eigen::Vector3d &p = ray_pair.first;
        const Eigen::Vector3d &q = ray_pair.second;
        const Eigen::Vector3d pp = sim.s * sim.R * p + sim.t;
        
        return (q-pp).squaredNorm();
    }

    int SimilarityEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<Similarity>* sims) const
    {
        Similarity sim;
        NonMinimalSolver(sample,&sim);
        sims->clear();
        sims->push_back(sim);
        return 1;
    }
    
    int SimilarityEstimator::NonMinimalSolver(const std::vector<int>& sample, Similarity*sim) const
    {
        const int N = sample.size();
        if ( N < min_sample_size() ) 
        {
            std::cout << "bad sample size: " << N << "\n";
            return 0;
        }
        
        Eigen::MatrixXd p(3,N);
        Eigen::MatrixXd q(3,N);
        for ( int i = 0; i < N; i++ )
        {
            const RayPair &corr = correspondences[sample[i]];
            p.col(i) = corr.first;
            q.col(i) = corr.second;
        }
        
        Eigen::Matrix4d T = Eigen::umeyama(p,q,true);
        
        Eigen::Matrix3d sR = T.block<3,3>(0,0);
        double s = std::cbrt(sR.determinant());
        Eigen::Matrix3d R = sR/s;
        Eigen::Vector3d t = T.block<3,1>(0,3);

        *sim = Similarity(s,R,t);
        
        //*sim = umeyama(p,q);

        /*
        std::cout << "results:  \n";
        for ( int i = 0; i < N; i++ )
        {
            //const Eigen::Vector3d &q = correspondences[sample[i]].first;
            //const Eigen::Vector3d &p = correspondences[sample[i]].second;
            //Eigen::Vector3d Tp = T.block<3,3>(0,0) * p + T.block<3,1>(0,3);
            //Eigen::Vector3d residuals = q - Tp;
            //double score = residuals.squaredNorm();
            double score = EvaluateModelOnPoint(*sim, sample[i]);
            std::cout << "score: " << score << "\n";
        }
        */

        
        return 1;
    }

    void SimilarityEstimator::LeastSquares(const std::vector<int>& sample, Similarity* sim) const
    {
        double s = sim->s;
        Eigen::Vector3d r = so3ln(sim->R);
        Eigen::Vector3d t = sim->t;
        std::vector<Eigen::Vector3d> p;
        std::vector<Eigen::Vector3d> q;
        
        ceres::Problem problem;

        for ( int ind : sample ) 
        {
            p.push_back(correspondences[ind].first);
            q.push_back(correspondences[ind].second);
        }

        for ( int i = 0; i < sample.size(); i++ )
        {
            SimilarityError *error = new SimilarityError;
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SimilarityError, 3, 1, 3, 3, 3, 3>(error);
            problem.AddResidualBlock(cost_function, NULL,
                &s, r.data(), t.data(),
                p[i].data(), q[i].data() );
            problem.SetParameterBlockConstant( p[i].data() );
            problem.SetParameterBlockConstant( q[i].data() );
        }

        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.max_num_consecutive_invalid_steps = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 16;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        sim->s = s;
        sim->R = so3exp(r);
        sim->t = t;
    }
}
