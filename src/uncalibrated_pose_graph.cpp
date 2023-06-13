
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

#include <sphericalsfm/so3.h>
#include <sphericalsfm/spherical_utils.h>

#include <sphericalsfm/uncalibrated_pose_graph.h>

namespace sphericalsfm {

    struct UncalibratedPoseGraphError
    {
        UncalibratedPoseGraphError(
            const Eigen::Matrix3d &_E_guess,
            const double _focal_guess,
            const double _rx,
            const double _ry,
            const double _thetaz,
            const double &_scale,
            const bool _inward ) : E_guess(_E_guess), focal_guess(_focal_guess), rx(_rx), ry(_ry), thetaz(_thetaz), scale(_scale), inward(_inward) { }
        
        template <typename T>
        bool operator()(const T* const r0,
                        const T* const r1,
                        const T* const f,
                        T* residuals) const
        {
            Eigen::Matrix<T,3,3> TT = Eigen::Matrix<T,3,3>::Identity();
            TT(0,0) = *f/focal_guess;
            TT(1,1) = *f/focal_guess;
            // convert fundamental matrix to essential matrix
            // E = K^T*F*K
            Eigen::Matrix<T,3,3> E_guess;
            for ( int i = 0; i < 3; i++ )
                for ( int j = 0; j < 3; j++ )
                    E_guess(i,j) = T(this->E_guess(i,j));
            Eigen::Matrix<T,3,3> E = TT * E_guess * TT;

            // normalize E(0,2) to equal -rx
            E = -E/E(0,2)*rx;
            
            // compute theta_xy
            T rx = T(this->rx);
            T ry = T(this->ry);
            T thetaz = T(this->thetaz);
            T sz = sin(thetaz);
            T sz2 = sz*sz;
            T cz = cos(thetaz);
            T cz2 = cz*cz;
            T rx2 = rx*rx;
            T rx3 = rx2*rx;
            T rx4 = rx2*rx2;
            T ry2 = ry*ry;
            T ry3 = ry2*ry;
            T ry4 = ry2*ry2;
            T s = (T(2)*E(0,0)*(- sz*rx2 + T(2)*cz*rx*ry + sz*ry2))/(rx4*sz2 + ry4*sz2 + E(0,0)*E(0,0) + T(4)*rx2*ry2*cz2 - T(2)*rx2*ry2*sz2 + T(4)*rx*ry3*cz*sz - T(4)*rx3*ry*cz*sz);
            T c = (rx4*sz2 + ry4*sz2 - E(0,0)*E(0,0) + T(4)*rx2*ry2*cz2 - T(2)*rx2*ry2*sz2 + T(4)*rx*ry3*cz*sz - T(4)*rx3*ry*cz*sz)/(rx4*sz2 + ry4*sz2 + E(0,0)*E(0,0) + T(4)*rx2*ry2*cz2 - T(2)*rx2*ry2*sz2 + T(4)*rx*ry3*cz*sz - T(4)*rx3*ry*cz*sz);
            T thetaxy = atan2(s,c);

            // compute Rxy and Rz
            T rxy[3] = { thetaxy*rx, thetaxy*ry, T(0) };
            T rz[3] = { T(0), T(0), thetaz };
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
        
        Eigen::Matrix3d E_guess;
        double focal_guess;
        double rx, ry, thetaz;
        double scale;
        bool inward;
    };

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

        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.03);
        for ( int i = 0; i < relative_rotations.size(); i++ )
        {
            Eigen::Matrix3d E;
            make_spherical_essential_matrix( relative_rotations[i].R, inward, E );
            
            // decompose rotation into xy and z rotations
            double rx, ry, thetaxy, thetaz;
            decompose_rotation( relative_rotations[i].R, rx, ry, thetaxy, thetaz );
            
            UncalibratedPoseGraphError *error = new UncalibratedPoseGraphError(E,focal_guess,rx,ry,thetaz,1/max_rel_rot_norm,inward);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<UncalibratedPoseGraphError, 3, 3, 3, 1>(error);
            problem.AddResidualBlock(cost_function,
                loss_function,
                data[relative_rotations[i].index0].data(),
                data[relative_rotations[i].index1].data(),
                &focal_length
            );
        }
        problem.SetParameterBlockConstant( data[0].data() );
        problem.SetParameterLowerBound( &focal_length, 0, min_focal );
        problem.SetParameterUpperBound( &focal_length, 0, max_focal );
    
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;//true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.FullReport() << "\n";
        if ( summary.termination_type == ceres::FAILURE )
        {
            std::cout << "error: ceres failed.\n";
            exit(1);
        }
        
        for ( int i = 0; i < rotations.size(); i++ ) rotations[i] = so3exp(data[i]);
        
        return summary.final_cost;
    }

}
