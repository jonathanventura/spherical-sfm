
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

#include <sphericalsfm/so3.h>

#include <sphericalsfm/rotation_averaging.h>

namespace sphericalsfm {

    struct RotationError
    {
        RotationError( const Eigen::Matrix3d &_meas ) : meas(_meas) { }
        
        template <typename T>
        bool operator()(const T* const r0,
                        const T* const r1,
                        T* residuals) const
        {
            Eigen::Matrix<T,3,3> R, R0, R1;
            for ( int i = 0; i < 3; i++ )
                for ( int j = 0; j < 3; j++ )
                    R(i,j) = T(meas(i,j));
            ceres::AngleAxisToRotationMatrix( r0, R0.data() );
            ceres::AngleAxisToRotationMatrix( r1, R1.data() );
            
            Eigen::Matrix<T,3,3> cycle = (R1 * R0.transpose()) * R.transpose();
            ceres::RotationMatrixToAngleAxis( cycle.data(), residuals );

            return true;
        }
        
        Eigen::Matrix3d meas;
    };

    void optimize_rotations( std::vector<Eigen::Matrix3d> &rotations, const std::vector<RelativeRotation> &relative_rotations )
    {
        std::vector<Eigen::Vector3d> data(rotations.size());
        for ( int i = 0; i < rotations.size(); i++ ) data[i] = so3ln(rotations[i]);
        
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.03);
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            std::cout << relative_rotations[i].index0 << "\n";
            std::cout << relative_rotations[i].index1 << "\n";
            std::cout << relative_rotations[i].R << "\n";
            RotationError *error = new RotationError(relative_rotations[i].R);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RotationError, 3, 3, 3>(error);
            problem.AddResidualBlock(cost_function,
                loss_function,
                data[relative_rotations[i].index0].data(),
                data[relative_rotations[i].index1].data()
            );
        }
    
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        if ( summary.termination_type == ceres::FAILURE )
        {
            std::cout << "error: ceres failed.\n";
            exit(1);
        }
        
        for ( int i = 0; i < rotations.size(); i++ ) rotations[i] = so3exp(data[i]);
    }

}
