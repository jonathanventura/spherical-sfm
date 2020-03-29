
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

#include <Polynomial/Polynomial.hpp>
using polynomial::Polynomial;

#include <sphericalsfm/spherical_fast_estimator.h>
#include <sphericalsfm/so3.h>

namespace sphericalsfm {
    int SphericalFastEstimator::sampleSize()
    {
        return 3;
    }

    double SphericalFastEstimator::score( RayPairList::iterator it )
    {
        const Eigen::Vector3d &u = it->first.head(3);
        const Eigen::Vector3d &v = it->second.head(3);
        const Eigen::Vector3d line = E * (u/u(2));
        const double d = v.dot( line );
        
        return (d*d) / (line[0]*line[0] + line[1]*line[1]);
    }

    void SphericalFastEstimator::chooseSolution( int soln )
    {
        E = Esolns[soln];
    }

    bool SphericalFastEstimator::canRefine()
    {
        return true;
    }

    int SphericalFastEstimator::compute( RayPairList::iterator begin, RayPairList::iterator end )
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
        
    //    Eigen::Matrix<double,6,3> B = A.jacobiSvd(Eigen::ComputeFullV).matrixV().block(0,3,6,3);
        Eigen::MatrixXd Q = A.transpose().colPivHouseholderQr().householderQ();
        Eigen::Matrix<double,6,3> B = Q.block(0,3,6,3);

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
        const double t66 = 2*B(1,0)*B(4,0);
        const double t16 = t15 - t66;
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
        const double t72 = 2*B(1,0)*B(4,1);
        const double t73 = 2*B(1,1)*B(4,0);
        const double t29 = t27 + t28 - t72 - t73;
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
        const double t45 = 2*B(1,1)*B(4,1);
        const double t42 = t41 - t45;
        const double t43 = 4*B(0,1)*B(0,2);
        const double t44 = 4*B(1,1)*B(1,2);
        const double t46 = 2*B(2,1)*B(2,2);
        const double t47 = 2*B(3,1)*B(3,2);
        const double t48 = 2*B(4,1)*B(4,2);
        const double t49 = 2*B(5,1)*B(5,2);
        const double t50 = t43 + t44 + t46 + t47 + t48 + t49;
        const double t51 = B(0,2)*B(0,2);
        const double t52 = 2*t51;
        const double t53 = B(1,2)*B(1,2);
        const double t54 = 2*t53;
        const double t55 = B(3,2)*B(3,2);
        const double t56 = 2*B(0,1)*B(5,2);
        const double t57 = 2*B(0,2)*B(5,1);
        const double t77 = 2*B(1,1)*B(4,2);
        const double t78 = 2*B(1,2)*B(4,1);
        const double t58 = t56 + t57 - t77 - t78;
        const double t59 = 4*B(3,1)*B(3,2);
        const double t60 = t43 + t44 + t59;
        const double t61 = B(0,1)*B(2,2);
        const double t62 = B(0,2)*B(2,1);
        const double t63 = t61 + t62;
        const double t64 = 4*B(0,0)*B(0,2);
        const double t65 = 4*B(1,0)*B(1,2);
        const double t67 = 2*B(2,0)*B(2,2);
        const double t68 = 2*B(3,0)*B(3,2);
        const double t69 = 2*B(4,0)*B(4,2);
        const double t70 = 2*B(5,0)*B(5,2);
        const double t71 = t64 + t65 + t67 + t68 + t69 + t70;
        const double t74 = 2*B(0,0)*B(5,2);
        const double t75 = 2*B(0,2)*B(5,0);
        const double t90 = 2*B(1,0)*B(4,2);
        const double t91 = 2*B(1,2)*B(4,0);
        const double t76 = t74 + t75 - t90 - t91;
        const double t79 = 4*B(3,0)*B(3,2);
        const double t80 = t64 + t65 + t79;
        const double t81 = B(0,0)*B(2,2);
        const double t82 = B(0,2)*B(2,0);
        const double t83 = t81 + t82;
        const double t84 = 2*t55;
        const double t85 = t52 + t54 + t84;
        const double t86 = B(2,2)*B(2,2);
        const double t87 = B(4,2)*B(4,2);
        const double t88 = B(5,2)*B(5,2);
        const double t89 = t52 + t54 + t55 + t86 + t87 + t88;
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
        const double t115 = 2*B(0,1)*B(4,2);
        const double t116 = 2*B(0,2)*B(4,1);
        const double t117 = 2*B(1,1)*B(5,2);
        const double t118 = 2*B(1,2)*B(5,1);
        const double t119 = t115 + t116 + t117 + t118;
        const double t120 = 4*B(4,1)*B(4,2);
        const double t121 = 4*B(5,1)*B(5,2);
        const double t122 = t120 + t121;
        const double t123 = 2*B(0,0)*B(4,2);
        const double t124 = 2*B(0,2)*B(4,0);
        const double t125 = 2*B(1,0)*B(5,2);
        const double t126 = 2*B(1,2)*B(5,0);
        const double t127 = t123 + t124 + t125 + t126;
        const double t128 = 4*B(4,0)*B(4,2);
        const double t129 = 4*B(5,0)*B(5,2);
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
        const double t140 = B(1,1)*B(2,2);
        const double t141 = B(1,2)*B(2,1);
        const double t142 = t140 + t141;
        const double t143 = B(1,0)*B(2,2);
        const double t144 = B(1,2)*B(2,0);
        const double t145 = t143 + t144;
        Eigen::Matrix<double,6,10> C;
        C << B(1,0)*t8 - B(1,0)*t12 - B(4,0)*t16 + 2*B(0,0)*B(2,0)*B(3,0), B(1,1)*t8 - B(1,1)*t12 - B(1,0)*t21 + B(1,0)*t31 - B(4,1)*t16 + 2*B(3,0)*t34 - B(4,0)*t29 + 2*B(0,0)*B(2,0)*B(3,1), B(1,1)*t31 - B(1,1)*t21 + B(1,0)*t36 - B(1,0)*t40 + 2*B(3,1)*t34 - B(4,1)*t29 - B(4,0)*t42 + 2*B(0,1)*B(2,1)*B(3,0), B(1,1)*t36 - B(1,1)*t40 - B(4,1)*t42 + 2*B(0,1)*B(2,1)*B(3,1), B(1,2)*t36 - B(1,2)*t40 - B(1,1)*t50 + B(1,1)*t60 - B(4,2)*t42 + 2*B(3,1)*t63 - B(4,1)*t58 + 2*B(0,1)*B(2,1)*B(3,2), B(1,2)*t60 - B(1,2)*t50 + 2*B(3,2)*t63 + B(1,1)*t85 - B(1,1)*t89 - B(4,2)*t58 - B(4,1)*t93 + 2*B(0,2)*B(2,2)*B(3,1), B(1,2)*t8 - B(1,2)*t12 - B(4,2)*t16 - B(1,0)*t71 + B(1,0)*t80 + 2*B(3,0)*t83 - B(4,0)*t76 + 2*B(0,0)*B(2,0)*B(3,2), B(1,2)*t31 - B(1,2)*t21 - B(1,0)*t50 + 2*B(3,2)*t34 + B(1,0)*t60 - B(4,2)*t29 - B(1,1)*t71 + B(1,1)*t80 + 2*B(3,0)*t63 - B(4,0)*t58 + 2*B(3,1)*t83 - B(4,1)*t76, B(1,2)*t80 - B(1,2)*t71 + B(1,0)*t85 - B(1,0)*t89 + 2*B(3,2)*t83 - B(4,2)*t76 - B(4,0)*t93 + 2*B(0,2)*B(2,2)*B(3,0), B(1,2)*t85 - B(1,2)*t89 - B(4,2)*t93 + 2*B(0,2)*B(2,2)*B(3,2),
        B(0,0)*t100 - B(4,0)*t12 - B(1,0)*t16 + B(4,0)*t97, B(0,1)*t100 - B(1,0)*t29 - B(4,1)*t12 - B(4,0)*t21 - B(1,1)*t16 + B(0,0)*t105 + B(4,1)*t97 + B(4,0)*t108, B(0,1)*t105 - B(1,0)*t42 - B(4,1)*t21 - B(4,0)*t40 - B(1,1)*t29 + B(0,0)*t114 + B(4,1)*t108 + B(4,0)*t111, B(0,1)*t114 - B(4,1)*t40 - B(1,1)*t42 + B(4,1)*t111, B(0,2)*t114 - B(1,1)*t58 - B(4,2)*t40 - B(4,1)*t50 - B(1,2)*t42 + B(0,1)*t119 + B(4,2)*t111 + B(4,1)*t122, B(0,2)*t119 - B(4,2)*t50 - B(1,1)*t93 - B(1,2)*t58 - B(4,1)*t89 + B(0,1)*t136 + B(4,2)*t122 + B(4,1)*t133, B(0,2)*t100 - B(4,2)*t12 - B(1,0)*t76 - B(1,2)*t16 - B(4,0)*t71 + B(0,0)*t127 + B(4,2)*t97 + B(4,0)*t130, B(0,2)*t105 - B(4,2)*t21 - B(1,0)*t58 - B(1,1)*t76 - B(4,0)*t50 - B(1,2)*t29 - B(4,1)*t71 + B(0,0)*t119 + B(0,1)*t127 + B(4,2)*t108 + B(4,0)*t122 + B(4,1)*t130, B(0,2)*t127 - B(1,0)*t93 - B(4,2)*t71 - B(1,2)*t76 - B(4,0)*t89 + B(0,0)*t136 + B(4,2)*t130 + B(4,0)*t133, B(0,2)*t136 - B(4,2)*t89 - B(1,2)*t93 + B(4,2)*t133,
        B(0,0)*t12 - B(0,0)*t8 - B(5,0)*t16 + 2*B(1,0)*B(2,0)*B(3,0), B(0,1)*t12 - B(0,1)*t8 + B(0,0)*t21 - B(0,0)*t31 - B(5,1)*t16 - B(5,0)*t29 + 2*B(3,0)*t139 + 2*B(1,0)*B(2,0)*B(3,1), B(0,1)*t21 - B(0,1)*t31 - B(0,0)*t36 + B(0,0)*t40 - B(5,1)*t29 - B(5,0)*t42 + 2*B(3,1)*t139 + 2*B(1,1)*B(2,1)*B(3,0), B(0,1)*t40 - B(0,1)*t36 - B(5,1)*t42 + 2*B(1,1)*B(2,1)*B(3,1), B(0,2)*t40 - B(0,2)*t36 + B(0,1)*t50 - B(0,1)*t60 - B(5,2)*t42 - B(5,1)*t58 + 2*B(3,1)*t142 + 2*B(1,1)*B(2,1)*B(3,2), B(0,2)*t50 - B(0,2)*t60 - B(0,1)*t85 + B(0,1)*t89 - B(5,2)*t58 - B(5,1)*t93 + 2*B(3,2)*t142 + 2*B(1,2)*B(2,2)*B(3,1), B(0,2)*t12 - B(0,2)*t8 - B(5,2)*t16 + B(0,0)*t71 - B(0,0)*t80 - B(5,0)*t76 + 2*B(3,0)*t145 + 2*B(1,0)*B(2,0)*B(3,2), B(0,2)*t21 - B(0,2)*t31 + B(0,0)*t50 - B(0,0)*t60 + B(0,1)*t71 - B(0,1)*t80 - B(5,2)*t29 - B(5,0)*t58 - B(5,1)*t76 + 2*B(3,2)*t139 + 2*B(3,0)*t142 + 2*B(3,1)*t145, B(0,2)*t71 - B(0,2)*t80 - B(0,0)*t85 + B(0,0)*t89 - B(5,2)*t76 - B(5,0)*t93 + 2*B(3,2)*t145 + 2*B(1,2)*B(2,2)*B(3,0), B(0,2)*t89 - B(0,2)*t85 - B(5,2)*t93 + 2*B(1,2)*B(2,2)*B(3,2),
        B(0,0)*t16 - B(5,0)*t12 + B(1,0)*t100 + B(5,0)*t97, B(0,1)*t16 + B(0,0)*t29 - B(5,1)*t12 - B(5,0)*t21 + B(1,1)*t100 + B(1,0)*t105 + B(5,1)*t97 + B(5,0)*t108, B(0,1)*t29 + B(0,0)*t42 - B(5,1)*t21 - B(5,0)*t40 + B(1,1)*t105 + B(1,0)*t114 + B(5,1)*t108 + B(5,0)*t111, B(0,1)*t42 - B(5,1)*t40 + B(1,1)*t114 + B(5,1)*t111, B(0,2)*t42 + B(0,1)*t58 - B(5,2)*t40 - B(5,1)*t50 + B(1,2)*t114 + B(1,1)*t119 + B(5,2)*t111 + B(5,1)*t122, B(0,2)*t58 + B(0,1)*t93 - B(5,2)*t50 + B(1,2)*t119 - B(5,1)*t89 + B(1,1)*t136 + B(5,2)*t122 + B(5,1)*t133, B(0,2)*t16 - B(5,2)*t12 + B(0,0)*t76 + B(1,2)*t100 - B(5,0)*t71 + B(1,0)*t127 + B(5,2)*t97 + B(5,0)*t130, B(0,2)*t29 + B(0,0)*t58 - B(5,2)*t21 + B(0,1)*t76 - B(5,0)*t50 + B(1,2)*t105 - B(5,1)*t71 + B(1,0)*t119 + B(1,1)*t127 + B(5,2)*t108 + B(5,0)*t122 + B(5,1)*t130, B(0,2)*t76 + B(0,0)*t93 - B(5,2)*t71 + B(1,2)*t127 - B(5,0)*t89 + B(1,0)*t136 + B(5,2)*t130 + B(5,0)*t133, B(0,2)*t93 - B(5,2)*t89 + B(1,2)*t136 + B(5,2)*t133,
        B(3,0)*t8 + 2*B(3,0)*t9 - B(3,0)*t12, B(3,1)*t8 + 2*B(3,1)*t9 - B(3,1)*t12 - B(3,0)*t21 + B(3,0)*t31 + 4*B(2,0)*B(2,1)*B(3,0), B(3,1)*t31 - B(3,1)*t21 + B(3,0)*t36 + 2*B(3,0)*t37 - B(3,0)*t40 + 4*B(2,0)*B(2,1)*B(3,1), B(3,1)*t36 + 2*B(3,1)*t37 - B(3,1)*t40, B(3,2)*t36 + 2*B(3,2)*t37 - B(3,2)*t40 - B(3,1)*t50 + B(3,1)*t60 + 4*B(2,1)*B(2,2)*B(3,1), B(3,2)*t60 - B(3,2)*t50 + B(3,1)*t85 + 2*B(3,1)*t86 - B(3,1)*t89 + 4*B(2,1)*B(2,2)*B(3,2), B(3,2)*t8 + 2*B(3,2)*t9 - B(3,2)*t12 - B(3,0)*t71 + B(3,0)*t80 + 4*B(2,0)*B(2,2)*B(3,0), B(3,2)*t31 - B(3,2)*t21 - B(3,0)*t50 + B(3,0)*t60 - B(3,1)*t71 + B(3,1)*t80 + 4*B(2,0)*B(2,1)*B(3,2) + 4*B(2,0)*B(2,2)*B(3,1) + 4*B(2,1)*B(2,2)*B(3,0), B(3,2)*t80 - B(3,2)*t71 + B(3,0)*t85 + 2*B(3,0)*t86 - B(3,0)*t89 + 4*B(2,0)*B(2,2)*B(3,2), B(3,2)*t85 + 2*B(3,2)*t86 - B(3,2)*t89,
        B(2,0)*t100 - B(3,0)*t16, B(2,1)*t100 - B(3,0)*t29 - B(3,1)*t16 + B(2,0)*t105, B(2,1)*t105 - B(3,0)*t42 - B(3,1)*t29 + B(2,0)*t114, B(2,1)*t114 - B(3,1)*t42, B(2,2)*t114 - B(3,1)*t58 - B(3,2)*t42 + B(2,1)*t119, B(2,2)*t119 - B(3,1)*t93 - B(3,2)*t58 + B(2,1)*t136, B(2,2)*t100 - B(3,0)*t76 - B(3,2)*t16 + B(2,0)*t127, B(2,2)*t105 - B(3,0)*t58 - B(3,1)*t76 - B(3,2)*t29 + B(2,0)*t119 + B(2,1)*t127, B(2,2)*t127 - B(3,0)*t93 - B(3,2)*t76 + B(2,0)*t136, B(2,2)*t136 - B(3,2)*t93;

        Eigen::Matrix<double,6,4> G( C.block(0,0,6,6).lu().solve(C.block(0,6,6,4)) );
    //    Eigen::Matrix<double,6,10> bigG( C.block(0,0,6,6).lu().solve(C) );
    //    std::cout << bigG << "\n\n";

        const double ypoly_coeffs[5] = {G(4,0)*G(5,1) - G(4,1)*G(5,0), G(3,1)*G(5,0) - G(3,0)*G(5,1) + G(4,0)*G(5,2) - G(4,2)*G(5,0), G(3,2)*G(5,0) - G(3,1)*G(4,0) + G(3,0)*(G(4,1) - G(5,2)), G(3,0)*(G(4,2) + G(4,1)*G(5,3) - G(4,3)*G(5,1)) + G(3,3)*(G(4,0)*G(5,1) - G(4,1)*G(5,0)) - G(3,1)*(G(4,0)*G(5,3) - G(4,3)*G(5,0)) - G(3,2)*G(4,0), G(3,3)*(G(4,0)*G(5,2) - G(4,2)*G(5,0)) - G(3,2)*(G(4,0)*G(5,3) - G(4,3)*G(5,0)) + G(3,0)*(G(4,2)*G(5,3) - G(4,3)*G(5,2))};
        const Polynomial<4> ypoly(ypoly_coeffs);
        
        std::vector<double> ysolns;
        ypoly.realRootsSturm( -10, 10, ysolns );
        
        int nsolns = 0;
        
        for ( int i = 0; i < ysolns.size(); i++ )
        {
            const double y = ysolns[i];
            const double ysq = y*y;
            const double ycu = ysq*y;
            
            Eigen::Matrix3d N;
            N <<
            G(3,0), G(3,2) + G(3,1)*y, G(3,3) + ycu,
            G(4,0), G(4,2) + G(4,1)*y, G(4,3) + ysq,
            G(5,0), G(5,2) + G(5,1)*y, G(5,3) + y;
            
            const double x = ( N(0,2)*N(1,0) - N(0,0)*N(1,2) ) / ( N(0,0)*N(1,1) - N(0,1)*N(1,0) );
            if ( std::isnan(x) ) continue;
            
            Eigen::Vector3d bsoln( x, y, 1 );
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

    static double my_triangulate1d( const Eigen::Matrix4d &rel_pose, const Eigen::Vector3d &u, const Eigen::Vector3d &v )
    {
        const Eigen::Matrix3d R = rel_pose.block<3,3>(0,0);
        const Eigen::Vector3d t = rel_pose.block<3,1>(0,3);
        
        const double inv_depth = -(v(2)*(R(0,0)*u(0) + R(0,1)*u(1) + R(0,2)*u(2)) - v(0)*(R(2,0)*u(0) + R(2,1)*u(1) + R(2,2)*u(2)))/(t(0)*v(2) - t(2)*v(0));
        
        return inv_depth;
    }

    static Eigen::Vector3d my_triangulateMidpoint( const Eigen::Matrix4d &rel_pose, const Eigen::Vector3d &u, const Eigen::Vector3d &v )
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

    void SphericalFastEstimator::decomposeE( bool inward, Eigen::Vector3d &r, Eigen::Vector3d &t )
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

/*
    void SphericalFastEstimator::decomposeE( bool inward,
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
            
            Eigen::Vector3d X1 = my_triangulateMidpoint(P1, u, v);
            Eigen::Vector3d X2 = my_triangulateMidpoint(P2, u, v);

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
*/
}

