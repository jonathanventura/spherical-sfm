
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

    static Eigen::Vector3d triangulateMidpoint( const Eigen::Matrix4d &rel_pose, const Eigen::Vector3d &u, const Eigen::Vector3d &v )
    {
        Eigen::Vector3d cu( 0, 0, 0 );
        Eigen::Vector3d cv( -rel_pose.block<3,3>(0,0).transpose() * rel_pose.block<3,1>(0,3) );
        
        Eigen::Matrix3d A;
        A <<
        u(0), -v(0), cu(0) - cv(0),
        u(1), -v(1), cu(1) - cv(1),
        u(2), -v(2), cu(2) - cv(2);
        
        const Eigen::Vector3d soln = A.jacobiSvd( Eigen::ComputeFullV ).matrixV().col(2);
        const double du = soln(0)/soln(2);
        const double dv = soln(1)/soln(2);
        
        const Eigen::Vector3d Xu = cu + u*du;
        const Eigen::Vector3d Xv = cv + v*dv;
        
        return (Xu+Xv)*0.5;
    }

    void decompose_spherical_essential_matrix( const Eigen::Matrix3d &E, bool inward,
                     RayPairList::iterator begin, RayPairList::iterator end, const std::vector<bool> &inliers,
                     Eigen::Vector3d &r, Eigen::Vector3d &t )
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
        
        Eigen::Matrix3d R1 = U*D*VT;
        Eigen::Matrix3d R2 = U*DT*VT;
        
        Eigen::Vector3d t1( R1(0,2), R1(1,2), (inward) ? R1(2,2)+1 : R1(2,2)-1 );
        Eigen::Vector3d t2( R2(0,2), R2(1,2), (inward) ? R2(2,2)+1 : R2(2,2)-1 );
        
        Eigen::Vector3d r1 = so3ln(R1);
        Eigen::Vector3d r2 = so3ln(R2);

        double r1test = r1.norm();
        double r2test = r2.norm();
        
        if ( r2test > M_PI/2 && r1test < M_PI/2 ) { r = r1; t = t1; return; }
        if ( r1test > M_PI/2 && r2test < M_PI/2 ) { r = r2; t = t2; return; }

        Eigen::Matrix4d P1( Eigen::Matrix4d::Identity() );
        Eigen::Matrix4d P2( Eigen::Matrix4d::Identity() );
        
        P1.block(0,0,3,3) = R1;
        P1.block(0,3,3,1) = t1;

        P2.block(0,0,3,3) = R2;
        P2.block(0,3,3,1) = t2;
        
        int ninfront1 = 0;
        int ninfront2 = 0;
        
        int i = 0;
        for ( RayPairList::iterator it = begin; it != end; it++,i++ )
        {
            if ( !inliers[i] ) continue;
            
            Eigen::Vector3d u = it->first.head(3);
            Eigen::Vector3d v = it->second.head(3);
            
            Eigen::Vector3d X1 = triangulateMidpoint(P1, u, v);
            Eigen::Vector3d X2 = triangulateMidpoint(P2, u, v);

            if ( X1(2) > 0 ) ninfront1++;
            if ( X2(2) > 0 ) ninfront2++;
        }
        
        if ( ninfront1 > ninfront2 )
        {
            r = so3ln(R1);
            t = t1;
        }
        else
        {
            r = so3ln(R2);
            t = t2;
        }
    }
}
