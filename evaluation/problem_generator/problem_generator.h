
#pragma once

#include <Eigen/Core>
#include <vector>
#include <sphericalsfm/ray.h>
#include <sphericalsfm/so3.h>

namespace problem_generator
{
    struct RelativePoseSolution
    {
        Eigen::Matrix3d E;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;

        double calc_frob_error(const Eigen::Matrix3d &Esoln ) const
        {
            Eigen::Matrix3d E_normalized = E/E.norm();
            Eigen::Matrix3d soln_E_normalized = Esoln/Esoln.norm();
            const double pos_error = (E_normalized-soln_E_normalized).norm();
            const double neg_error = (E_normalized+soln_E_normalized).norm();
            return std::min(pos_error,neg_error);
        }

        double calc_rot_error(const Eigen::Matrix3d &Rsoln ) const
        {
            return sphericalsfm::so3ln(Rsoln*R.transpose()).norm();
        }

        double calc_trans_error(const Eigen::Vector3d &tsoln ) const
        {
            const Eigen::Vector3d t_norm = t/t.norm();
            const Eigen::Vector3d soln_t_norm = tsoln/tsoln.norm();
            const double pos_err = acos(std::max(std::min(t_norm.dot(soln_t_norm),1.),-1.));
            const double neg_err = acos(std::max(std::min(-t_norm.dot(soln_t_norm),1.),-1.));
            return std::min(pos_err,neg_err);
        }
    };

    struct RelativePoseProblem
    {
        RelativePoseSolution soln; // ground truth solution
        sphericalsfm::RayPairList correspondences;
    };

    class ProblemGenerator
    {
        double point_noise;
    public:
        ProblemGenerator( double _point_noise )
        : point_noise( _point_noise )
        {
            
        }
        
        RelativePoseProblem make_random_problem( int num_corr, bool inward, double rotation = -1 );
    };
}
