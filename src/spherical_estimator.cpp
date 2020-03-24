
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>

#include <sphericalsfm/spherical_estimator.h>
#include <sphericalsfm/so3.h>
#include <iostream>
namespace sphericalsfm {
    int SphericalEstimator::sampleSize()
    {
        return 3;
    }

    double SphericalEstimator::score( RayPairList::iterator it )
    {
        const Eigen::Vector3d &u = it->first.head(3);
        const Eigen::Vector3d &v = it->second.head(3);
        const Eigen::Vector3d line = E * (u/u(2));
        const double d = v.dot( line );
        
        return (d*d) / (line[0]*line[0] + line[1]*line[1]);
    }

    void SphericalEstimator::chooseSolution( int soln )
    {
        E = Esolns[soln];
    }

    bool SphericalEstimator::canRefine()
    {
        return true;
    }

    int SphericalEstimator::compute( RayPairList::iterator begin, RayPairList::iterator end )
    {
        int N = std::distance(begin,end);
        
        Eigen::MatrixXd A(N,6);

        int i = 0;
        for ( RayPairList::iterator it = begin; it != end; it++,i++ )
        {
            const Eigen::Vector3d u = it->first.head(3);
            const Eigen::Vector3d v = it->second.head(3);
        
            A.row(i) << u(0)*v(0) - u(1)*v(1), u(0)*v(1) + u(1)*v(0), u(2)*v(0), u(2)*v(1), u(0)*v(2), u(1)*v(2);
        }
        
        Eigen::MatrixXd Q = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixV();
//        std::cout << " V size: " << Q.rows() << " " << Q.cols() << "\n";
        Eigen::Matrix<double,6,3> B = Q.block(0,3,6,3);
        
        // replace with QR(A.') --> last rows of Q are nullspace
//        Eigen::MatrixXd Q = A.transpose().colPivHouseholderQr().householderQ();
//        Eigen::Matrix<double,6,3> B = Q.block(0,3,6,3);

        const double t2 = B(0,0)*B(0,0);
        const double t3 = 2*t2;
        const double t4 = B(1,0)*B(1,0);
        const double t5 = 2*t4;
        const double t6 = B(3,0)*B(3,0);
        const double t7 = 2*t6;
        const double t8 = t3 + t5 + t7;
        const double t9 = B(2,0)*B(2,0);
        const double t10 = B(4,0)*B(4,0);
        const double t11 = B(5,0)*B(5,0);
        const double t12 = t3 + t5 + t6 + t9 + t10 + t11;
        const double t13 = 4*B(0,0)*B(0,1);
        const double t14 = 4*B(1,0)*B(1,1);
        const double t15 = 2*B(0,0)*B(5,0);
        const double t45 = 2*B(1,0)*B(4,0);
        const double t16 = t15 - t45;
        const double t17 = 2*B(2,0)*B(2,1);
        const double t18 = 2*B(3,0)*B(3,1);
        const double t19 = 2*B(4,0)*B(4,1);
        const double t20 = 2*B(5,0)*B(5,1);
        const double t21 = t13 + t14 + t17 + t18 + t19 + t20;
        const double t22 = B(0,1)*B(0,1);
        const double t23 = 2*t22;
        const double t24 = B(1,1)*B(1,1);
        const double t25 = 2*t24;
        const double t26 = B(3,1)*B(3,1);
        const double t27 = 2*B(0,0)*B(5,1);
        const double t28 = 2*B(0,1)*B(5,0);
        const double t51 = 2*B(1,0)*B(4,1);
        const double t52 = 2*B(1,1)*B(4,0);
        const double t29 = t27 + t28 - t51 - t52;
        const double t30 = 4*B(3,0)*B(3,1);
        const double t31 = t13 + t14 + t30;
        const double t32 = B(0,0)*B(2,1);
        const double t33 = B(0,1)*B(2,0);
        const double t34 = t32 + t33;
        const double t35 = 2*t26;
        const double t36 = t23 + t25 + t35;
        const double t37 = B(2,1)*B(2,1);
        const double t38 = B(4,1)*B(4,1);
        const double t39 = B(5,1)*B(5,1);
        const double t40 = t23 + t25 + t26 + t37 + t38 + t39;
        const double t41 = 2*B(0,1)*B(5,1);
        const double t76 = 2*B(1,1)*B(4,1);
        const double t42 = t41 - t76;
        const double t43 = 4*B(0,0)*B(0,2);
        const double t44 = 4*B(1,0)*B(1,2);
        const double t46 = 2*B(2,0)*B(2,2);
        const double t47 = 2*B(3,0)*B(3,2);
        const double t48 = 2*B(4,0)*B(4,2);
        const double t49 = 2*B(5,0)*B(5,2);
        const double t50 = t43 + t44 + t46 + t47 + t48 + t49;
        const double t53 = 2*B(0,0)*B(5,2);
        const double t54 = 2*B(0,2)*B(5,0);
        const double t82 = 2*B(1,0)*B(4,2);
        const double t83 = 2*B(1,2)*B(4,0);
        const double t55 = t53 + t54 - t82 - t83;
        const double t56 = 4*B(3,0)*B(3,2);
        const double t57 = t43 + t44 + t56;
        const double t58 = 4*B(0,1)*B(0,2);
        const double t59 = 4*B(1,1)*B(1,2);
        const double t60 = B(0,0)*B(2,2);
        const double t61 = B(0,2)*B(2,0);
        const double t62 = t60 + t61;
        const double t63 = 2*B(2,1)*B(2,2);
        const double t64 = 2*B(3,1)*B(3,2);
        const double t65 = 2*B(4,1)*B(4,2);
        const double t66 = 2*B(5,1)*B(5,2);
        const double t67 = t58 + t59 + t63 + t64 + t65 + t66;
        const double t68 = 2*B(0,1)*B(5,2);
        const double t69 = 2*B(0,2)*B(5,1);
        const double t90 = 2*B(1,1)*B(4,2);
        const double t91 = 2*B(1,2)*B(4,1);
        const double t70 = t68 + t69 - t90 - t91;
        const double t71 = 4*B(3,1)*B(3,2);
        const double t72 = t58 + t59 + t71;
        const double t73 = B(0,1)*B(2,2);
        const double t74 = B(0,2)*B(2,1);
        const double t75 = t73 + t74;
        const double t77 = B(0,2)*B(0,2);
        const double t78 = 2*t77;
        const double t79 = B(1,2)*B(1,2);
        const double t80 = 2*t79;
        const double t81 = B(3,2)*B(3,2);
        const double t84 = 2*t81;
        const double t85 = t78 + t80 + t84;
        const double t86 = B(2,2)*B(2,2);
        const double t87 = B(4,2)*B(4,2);
        const double t88 = B(5,2)*B(5,2);
        const double t89 = t78 + t80 + t81 + t86 + t87 + t88;
        const double t92 = 2*B(0,2)*B(5,2);
        const double t94 = 2*B(1,2)*B(4,2);
        const double t93 = t92 - t94;
        const double t95 = 2*t10;
        const double t96 = 2*t11;
        const double t97 = t95 + t96;
        const double t98 = 2*B(0,0)*B(4,0);
        const double t99 = 2*B(1,0)*B(5,0);
        const double t100 = t98 + t99;
        const double t101 = 2*B(0,0)*B(4,1);
        const double t102 = 2*B(0,1)*B(4,0);
        const double t103 = 2*B(1,0)*B(5,1);
        const double t104 = 2*B(1,1)*B(5,0);
        const double t105 = t101 + t102 + t103 + t104;
        const double t106 = 4*B(4,0)*B(4,1);
        const double t107 = 4*B(5,0)*B(5,1);
        const double t108 = t106 + t107;
        const double t109 = 2*t38;
        const double t110 = 2*t39;
        const double t111 = t109 + t110;
        const double t112 = 2*B(0,1)*B(4,1);
        const double t113 = 2*B(1,1)*B(5,1);
        const double t114 = t112 + t113;
        const double t115 = 2*B(0,0)*B(4,2);
        const double t116 = 2*B(0,2)*B(4,0);
        const double t117 = 2*B(1,0)*B(5,2);
        const double t118 = 2*B(1,2)*B(5,0);
        const double t119 = t115 + t116 + t117 + t118;
        const double t120 = 4*B(4,0)*B(4,2);
        const double t121 = 4*B(5,0)*B(5,2);
        const double t122 = t120 + t121;
        const double t123 = 2*B(0,1)*B(4,2);
        const double t124 = 2*B(0,2)*B(4,1);
        const double t125 = 2*B(1,1)*B(5,2);
        const double t126 = 2*B(1,2)*B(5,1);
        const double t127 = t123 + t124 + t125 + t126;
        const double t128 = 4*B(4,1)*B(4,2);
        const double t129 = 4*B(5,1)*B(5,2);
        const double t130 = t128 + t129;
        const double t131 = 2*t87;
        const double t132 = 2*t88;
        const double t133 = t131 + t132;
        const double t134 = 2*B(0,2)*B(4,2);
        const double t135 = 2*B(1,2)*B(5,2);
        const double t136 = t134 + t135;
        const double t137 = B(1,0)*B(2,1);
        const double t138 = B(1,1)*B(2,0);
        const double t139 = t137 + t138;
        const double t140 = B(1,0)*B(2,2);
        const double t141 = B(1,2)*B(2,0);
        const double t142 = t140 + t141;
        const double t143 = B(1,1)*B(2,2);
        const double t144 = B(1,2)*B(2,1);
        const double t145 = t143 + t144;
        Eigen::Matrix<double,6,10> C;
        C << B(1,0)*t8 - B(1,0)*t12 - B(4,0)*t16 + 2*B(0,0)*B(2,0)*B(3,0), B(1,1)*t8 - B(1,1)*t12 - B(1,0)*t21 + B(1,0)*t31 - B(4,1)*t16 + 2*B(3,0)*t34 - B(4,0)*t29 + 2*B(0,0)*B(2,0)*B(3,1), B(1,1)*t31 - B(1,1)*t21 + B(1,0)*t36 - B(1,0)*t40 + 2*B(3,1)*t34 - B(4,1)*t29 - B(4,0)*t42 + 2*B(0,1)*B(2,1)*B(3,0), B(1,1)*t36 - B(1,1)*t40 - B(4,1)*t42 + 2*B(0,1)*B(2,1)*B(3,1), B(1,2)*t8 - B(1,2)*t12 - B(4,2)*t16 - B(1,0)*t50 + B(1,0)*t57 + 2*B(3,0)*t62 - B(4,0)*t55 + 2*B(0,0)*B(2,0)*B(3,2), B(1,2)*t31 - B(1,2)*t21 - B(1,1)*t50 + 2*B(3,2)*t34 + B(1,1)*t57 - B(4,2)*t29 - B(1,0)*t67 + B(1,0)*t72 + 2*B(3,1)*t62 - B(4,1)*t55 + 2*B(3,0)*t75 - B(4,0)*t70, B(1,2)*t36 - B(1,2)*t40 - B(1,1)*t67 + B(1,1)*t72 - B(4,2)*t42 + 2*B(3,1)*t75 - B(4,1)*t70 + 2*B(0,1)*B(2,1)*B(3,2), B(1,2)*t57 - B(1,2)*t50 + 2*B(3,2)*t62 + B(1,0)*t85 - B(4,2)*t55 - B(1,0)*t89 - B(4,0)*t93 + 2*B(0,2)*B(2,2)*B(3,0), B(1,2)*t72 - B(1,2)*t67 + B(1,1)*t85 - B(1,1)*t89 + 2*B(3,2)*t75 - B(4,2)*t70 - B(4,1)*t93 + 2*B(0,2)*B(2,2)*B(3,1), B(1,2)*t85 - B(1,2)*t89 - B(4,2)*t93 + 2*B(0,2)*B(2,2)*B(3,2),
        B(0,0)*t100 - B(4,0)*t12 - B(1,0)*t16 + B(4,0)*t97, B(0,1)*t100 - B(1,0)*t29 - B(4,1)*t12 - B(4,0)*t21 - B(1,1)*t16 + B(0,0)*t105 + B(4,1)*t97 + B(4,0)*t108, B(0,1)*t105 - B(1,0)*t42 - B(4,1)*t21 - B(4,0)*t40 - B(1,1)*t29 + B(0,0)*t114 + B(4,1)*t108 + B(4,0)*t111, B(0,1)*t114 - B(4,1)*t40 - B(1,1)*t42 + B(4,1)*t111, B(0,2)*t100 - B(4,2)*t12 - B(1,0)*t55 - B(4,0)*t50 - B(1,2)*t16 + B(0,0)*t119 + B(4,2)*t97 + B(4,0)*t122, B(0,2)*t105 - B(4,2)*t21 - B(1,1)*t55 - B(1,0)*t70 - B(4,1)*t50 - B(1,2)*t29 - B(4,0)*t67 + B(0,1)*t119 + B(0,0)*t127 + B(4,2)*t108 + B(4,1)*t122 + B(4,0)*t130, B(0,2)*t114 - B(1,1)*t70 - B(4,2)*t40 - B(4,1)*t67 - B(1,2)*t42 + B(0,1)*t127 + B(4,2)*t111 + B(4,1)*t130, B(0,2)*t119 - B(4,2)*t50 - B(1,0)*t93 - B(1,2)*t55 - B(4,0)*t89 + B(0,0)*t136 + B(4,2)*t122 + B(4,0)*t133, B(0,2)*t127 - B(1,1)*t93 - B(4,2)*t67 - B(1,2)*t70 - B(4,1)*t89 + B(0,1)*t136 + B(4,2)*t130 + B(4,1)*t133, B(0,2)*t136 - B(4,2)*t89 - B(1,2)*t93 + B(4,2)*t133,
        B(0,0)*t12 - B(0,0)*t8 - B(5,0)*t16 + 2*B(1,0)*B(2,0)*B(3,0), B(0,1)*t12 - B(0,1)*t8 + B(0,0)*t21 - B(0,0)*t31 - B(5,1)*t16 - B(5,0)*t29 + 2*B(3,0)*t139 + 2*B(1,0)*B(2,0)*B(3,1), B(0,1)*t21 - B(0,1)*t31 - B(0,0)*t36 + B(0,0)*t40 - B(5,1)*t29 - B(5,0)*t42 + 2*B(3,1)*t139 + 2*B(1,1)*B(2,1)*B(3,0), B(0,1)*t40 - B(0,1)*t36 - B(5,1)*t42 + 2*B(1,1)*B(2,1)*B(3,1), B(0,2)*t12 - B(0,2)*t8 + B(0,0)*t50 - B(0,0)*t57 - B(5,2)*t16 - B(5,0)*t55 + 2*B(3,0)*t142 + 2*B(1,0)*B(2,0)*B(3,2), B(0,2)*t21 - B(0,2)*t31 + B(0,1)*t50 - B(0,1)*t57 + B(0,0)*t67 - B(0,0)*t72 - B(5,2)*t29 - B(5,1)*t55 - B(5,0)*t70 + 2*B(3,2)*t139 + 2*B(3,1)*t142 + 2*B(3,0)*t145, B(0,2)*t40 - B(0,2)*t36 + B(0,1)*t67 - B(0,1)*t72 - B(5,2)*t42 - B(5,1)*t70 + 2*B(3,1)*t145 + 2*B(1,1)*B(2,1)*B(3,2), B(0,2)*t50 - B(0,2)*t57 - B(0,0)*t85 + B(0,0)*t89 - B(5,2)*t55 - B(5,0)*t93 + 2*B(3,2)*t142 + 2*B(1,2)*B(2,2)*B(3,0), B(0,2)*t67 - B(0,2)*t72 - B(0,1)*t85 + B(0,1)*t89 - B(5,2)*t70 - B(5,1)*t93 + 2*B(3,2)*t145 + 2*B(1,2)*B(2,2)*B(3,1), B(0,2)*t89 - B(0,2)*t85 - B(5,2)*t93 + 2*B(1,2)*B(2,2)*B(3,2),
        B(0,0)*t16 - B(5,0)*t12 + B(1,0)*t100 + B(5,0)*t97, B(0,1)*t16 + B(0,0)*t29 - B(5,1)*t12 - B(5,0)*t21 + B(1,1)*t100 + B(1,0)*t105 + B(5,1)*t97 + B(5,0)*t108, B(0,1)*t29 + B(0,0)*t42 - B(5,1)*t21 - B(5,0)*t40 + B(1,1)*t105 + B(1,0)*t114 + B(5,1)*t108 + B(5,0)*t111, B(0,1)*t42 - B(5,1)*t40 + B(1,1)*t114 + B(5,1)*t111, B(0,2)*t16 + B(0,0)*t55 - B(5,2)*t12 - B(5,0)*t50 + B(1,2)*t100 + B(1,0)*t119 + B(5,2)*t97 + B(5,0)*t122, B(0,2)*t29 + B(0,1)*t55 + B(0,0)*t70 - B(5,2)*t21 - B(5,1)*t50 + B(1,2)*t105 - B(5,0)*t67 + B(1,1)*t119 + B(1,0)*t127 + B(5,2)*t108 + B(5,1)*t122 + B(5,0)*t130, B(0,2)*t42 + B(0,1)*t70 - B(5,2)*t40 - B(5,1)*t67 + B(1,2)*t114 + B(1,1)*t127 + B(5,2)*t111 + B(5,1)*t130, B(0,2)*t55 + B(0,0)*t93 - B(5,2)*t50 + B(1,2)*t119 - B(5,0)*t89 + B(1,0)*t136 + B(5,2)*t122 + B(5,0)*t133, B(0,2)*t70 + B(0,1)*t93 - B(5,2)*t67 + B(1,2)*t127 - B(5,1)*t89 + B(1,1)*t136 + B(5,2)*t130 + B(5,1)*t133, B(0,2)*t93 - B(5,2)*t89 + B(1,2)*t136 + B(5,2)*t133,
        B(3,0)*t8 + 2*B(3,0)*t9 - B(3,0)*t12, B(3,1)*t8 + 2*B(3,1)*t9 - B(3,1)*t12 - B(3,0)*t21 + B(3,0)*t31 + 4*B(2,0)*B(2,1)*B(3,0), B(3,1)*t31 - B(3,1)*t21 + B(3,0)*t36 + 2*B(3,0)*t37 - B(3,0)*t40 + 4*B(2,0)*B(2,1)*B(3,1), B(3,1)*t36 + 2*B(3,1)*t37 - B(3,1)*t40, B(3,2)*t8 + 2*B(3,2)*t9 - B(3,2)*t12 - B(3,0)*t50 + B(3,0)*t57 + 4*B(2,0)*B(2,2)*B(3,0), B(3,2)*t31 - B(3,2)*t21 - B(3,1)*t50 + B(3,1)*t57 - B(3,0)*t67 + B(3,0)*t72 + 4*B(2,0)*B(2,1)*B(3,2) + 4*B(2,0)*B(2,2)*B(3,1) + 4*B(2,1)*B(2,2)*B(3,0), B(3,2)*t36 + 2*B(3,2)*t37 - B(3,2)*t40 - B(3,1)*t67 + B(3,1)*t72 + 4*B(2,1)*B(2,2)*B(3,1), B(3,2)*t57 - B(3,2)*t50 + B(3,0)*t85 + 2*B(3,0)*t86 - B(3,0)*t89 + 4*B(2,0)*B(2,2)*B(3,2), B(3,2)*t72 - B(3,2)*t67 + B(3,1)*t85 + 2*B(3,1)*t86 - B(3,1)*t89 + 4*B(2,1)*B(2,2)*B(3,2), B(3,2)*t85 + 2*B(3,2)*t86 - B(3,2)*t89,
        B(2,0)*t100 - B(3,0)*t16, B(2,1)*t100 - B(3,0)*t29 - B(3,1)*t16 + B(2,0)*t105, B(2,1)*t105 - B(3,0)*t42 - B(3,1)*t29 + B(2,0)*t114, B(2,1)*t114 - B(3,1)*t42, B(2,2)*t100 - B(3,0)*t55 - B(3,2)*t16 + B(2,0)*t119, B(2,2)*t105 - B(3,1)*t55 - B(3,0)*t70 - B(3,2)*t29 + B(2,1)*t119 + B(2,0)*t127, B(2,2)*t114 - B(3,1)*t70 - B(3,2)*t42 + B(2,1)*t127, B(2,2)*t119 - B(3,0)*t93 - B(3,2)*t55 + B(2,0)*t136, B(2,2)*t127 - B(3,1)*t93 - B(3,2)*t70 + B(2,1)*t136, B(2,2)*t136 - B(3,2)*t93;
        
        Eigen::Matrix<double,6,4> G( C.block(0,0,6,6).lu().solve(C.block(0,6,6,4)) );

        Eigen::Matrix4d M( Eigen::Matrix4d::Zero() );
        M.row(0) = -G.row(2);
        M.row(1) = -G.row(4);
        M.row(2) = -G.row(5);
        M(3,1) = 1;

        Eigen::EigenSolver<Eigen::Matrix4d> eigM(M);
        Eigen::EigenSolver<Eigen::Matrix4d>::EigenvectorsType V = eigM.eigenvectors();
        Eigen::EigenSolver<Eigen::Matrix4d>::EigenvalueType evalues = eigM.eigenvalues();
        
        int nsolns = 0;
        
        for ( int i = 0; i < 4; i++ )
        {
            if ( fabs(evalues(i).imag()) > 1e-12 ) continue;
            
            Eigen::Vector3d bsoln( V(1,i).real(), V(2,i).real(), V(3,i).real() );
            Eigen::Matrix<double,6,1> psoln( B*bsoln );
            
            Eigen::Matrix3d Esoln;
            Esoln <<
            psoln(0), psoln(1), psoln(2),
            psoln(1), -psoln(0), psoln(3),
            psoln(4), psoln(5), 0;
            
            Esoln /= Esoln.norm();
            
            Esolns[nsolns++] = Esoln;
        }
        
        return nsolns;
    }

    void SphericalEstimator::decomposeE( bool inward,
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
