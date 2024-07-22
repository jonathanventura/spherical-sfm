
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

#include <sphericalsfm/so3.h>
#include <sphericalsfm/spherical_utils.h>

#include <sphericalsfm/uncalibrated_pose_graph.h>

namespace sphericalsfm {

    static void decompose_rotation( const Eigen::Matrix3d &R, double &rx, double &ry, double &thetaxy, double &thetaz )
    {
        Eigen::Vector3d Z = R.col(2);
        Z = Z/Z.norm();
        Eigen::Vector3d out(0,0,1);
        Eigen::Vector3d axis = out.cross(Z);
        Eigen::Vector3d rxy = axis/axis.norm();
        thetaxy = acos(Z.dot(out));
        Eigen::Matrix3d Rxy = so3exp(thetaxy*rxy);
        rx = rxy(0);
        ry = rxy(1);

        Eigen::Matrix3d Rz = Rxy.transpose()*R;
        Eigen::Vector3d rz = so3ln(Rz);
        thetaz = rz(2);
    }

    struct UncalibratedPoseGraphError
    {
        UncalibratedPoseGraphError(
            const Eigen::Vector3d &_r,
            const double &_scale )
            : scale( _scale )
        {
            decompose_rotation( so3exp(_r), rx, ry, thetaxy, thetaz );
        }
        
        template <typename T>
        bool operator()(const T* const r0,
                        const T* const r1,
                        const T* const f,
                        T* residuals) const
        {
            const T fsq = (*f)*(*f);
            const T num = T(2)*(*f)*sin(thetaxy);
            const T den = (T(1)+fsq)*cos(thetaxy)+(T(1)-fsq);
            const T thetaxyp = atan2(num,den);

            // compute Rxy and Rz
            const T rxy[3] = { thetaxyp*T(rx), thetaxyp*T(ry), T(0) };
            const T rz[3] = { T(0), T(0), T(thetaz) };
            Eigen::Matrix<T,3,3> Rxy, Rz;
            ceres::AngleAxisToRotationMatrix( rxy, Rxy.data() );
            ceres::AngleAxisToRotationMatrix( rz, Rz.data() );

            // compute R = Rxy * Rz
            Eigen::Matrix<T,3,3> R, R0, R1;
            R = Rxy*Rz;
            ceres::AngleAxisToRotationMatrix( r0, R0.data() );
            ceres::AngleAxisToRotationMatrix( r1, R1.data() );
            
            // compute rotation error
            Eigen::Matrix<T,3,3> cycle = (R1 * R0.transpose()) * R.transpose();
            ceres::RotationMatrixToAngleAxis( cycle.data(), residuals );
            residuals[0] *= scale;
            residuals[1] *= scale;
            residuals[2] *= scale;

            return true;
        }
        
        double rx, ry, thetaxy, thetaz;
        double scale;
    };

    struct PoseGraphError
    {
        PoseGraphError(
            const Eigen::Vector3d &_r,
            const double &_scale )
            : r(_r), scale( _scale )
        {

        }
        
        template <typename T>
        bool operator()(const T* const r0,
                        const T* const r1,
                        T* residuals) const
        {
            T myr[3] = { T(r(0)), T(r(1)), T(r(2)) };
            Eigen::Matrix<T,3,3> R, R0, R1;
            ceres::AngleAxisToRotationMatrix( myr, R.data() );
            ceres::AngleAxisToRotationMatrix( r0, R0.data() );
            ceres::AngleAxisToRotationMatrix( r1, R1.data() );
            
            // compute rotation error
            Eigen::Matrix<T,3,3> cycle = (R1 * R0.transpose()) * R.transpose();
            ceres::RotationMatrixToAngleAxis( cycle.data(), residuals );
            residuals[0] *= scale;
            residuals[1] *= scale;
            residuals[2] *= scale;

            return true;
        }
        
        Eigen::Vector3d r;
        double scale;
    };

    double get_cost( 
        std::vector<Eigen::Matrix3d> &rotations, 
        const std::vector<RelativeRotation> &relative_rotations )
    {
        std::vector<Eigen::Vector3d> data(rotations.size());
        for ( int i = 0; i < rotations.size(); i++ ) data[i] = so3ln(rotations[i]);
        
        double max_rel_rot_norm = 0;
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            double rot_norm = so3ln(relative_rotations[i].R).norm();
            if ( rot_norm > max_rel_rot_norm ) max_rel_rot_norm = rot_norm;
        }

        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.03);
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            PoseGraphError *error = new PoseGraphError(so3ln(relative_rotations[i].R),1/max_rel_rot_norm);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<PoseGraphError, 3, 3, 3>(error);
            problem.AddResidualBlock(cost_function,
                loss_function,
                data[relative_rotations[i].index0].data(),
                data[relative_rotations[i].index1].data()
            );
        }
        double cost;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
        return cost;
    }

    double optimize_rotations_and_focal_length( 
        std::vector<Eigen::Matrix3d> &rotations, 
        const std::vector<RelativeRotation> &relative_rotations, 
        double &focal_length,
        const double min_focal, const double max_focal,
        const double focal_guess, 
        bool inward )
    {
        std::vector<Eigen::Vector3d> data(rotations.size());
        for ( int i = 0; i < rotations.size(); i++ ) data[i] = so3ln(rotations[i]);
        
        double max_rel_rot_norm = 0;
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            double rot_norm = so3ln(relative_rotations[i].R).norm();
            if ( rot_norm > max_rel_rot_norm ) max_rel_rot_norm = rot_norm;
        }
        
        double focal_mult = 1;

        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.03);
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            UncalibratedPoseGraphError *error = new UncalibratedPoseGraphError(so3ln(relative_rotations[i].R),1/max_rel_rot_norm);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<UncalibratedPoseGraphError, 3, 3, 3, 1>(error);
            problem.AddResidualBlock(cost_function,
                loss_function,
                data[relative_rotations[i].index0].data(),
                data[relative_rotations[i].index1].data(),
                &focal_mult
            );
        }
        problem.SetParameterBlockConstant( data[0].data() );
        problem.SetParameterLowerBound( &focal_mult, 0, min_focal/focal_length );
        problem.SetParameterUpperBound( &focal_mult, 0, max_focal/focal_length );

        double cost;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
    
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
        focal_length *= focal_mult;
        
        return summary.final_cost;
    }

}
