
#include <Eigen/Jacobi>
#include <Eigen/SVD>

#include <sphericalsfm/so3.h>
#include <sphericalsfm/spherical_utils.h>

namespace sphericalsfm {
    void make_spherical_essential_matrix( const Eigen::Matrix3d &R, bool inward, Eigen::Matrix3d &E )
    {
        Eigen::Vector3d t( R(0,2), R(1,2), R(2,2)-1 );
        if ( inward ) t = -t;
        E = skew3(t)*R;
    }

    void decompose_spherical_essential_matrix( const Eigen::Matrix3d &E, bool inward, Eigen::Vector3d &r, Eigen::Vector3d &t )
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E,Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        Eigen::Matrix3d U = svdE.matrixU();
        Eigen::Matrix3d V = svdE.matrixV();
        
        // from theia sfm
        if (U.determinant() < 0) {
            U.col(2) *= -1.0;
        }
        
        if (V.determinant() < 0) {
            V.col(2) *= -1.0;
        }
        
        Eigen::Matrix3d D;
        D <<
        0,1,0,
        -1,0,0,
        0,0,1;
        
        Eigen::Matrix3d DT;
        DT <<
        0,-1,0,
        1,0,0,
        0,0,1;
        
        Eigen::Matrix3d VT = V.transpose().eval();
        
        Eigen::Vector3d tu = U.col(2);
        
        Eigen::Matrix3d R1 = U*D*VT;
        Eigen::Matrix3d R2 = U*DT*VT;
        
        Eigen::Vector3d t1( R1(0,2), R1(1,2), R1(2,2)-1 );
        Eigen::Vector3d t2( R2(0,2), R2(1,2), R2(2,2)-1 );
        
        if ( inward ) { t1 = -t1; t2 = -t2; }
        
        Eigen::Vector3d myt1 = t1/t1.norm();
        Eigen::Vector3d myt2 = t2/t2.norm();
        
        Eigen::Vector3d r1 = so3ln(R1);
        Eigen::Vector3d r2 = so3ln(R2);
        
        double score1 = fabs(myt1.dot(tu));
        double score2 = fabs(myt2.dot(tu));
        
        if ( score1 > score2 ) { r = r1; t = t1; }
        else { r = r2; t = t2; }
    }
}
