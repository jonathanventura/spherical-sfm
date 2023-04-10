
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Jacobi>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>

#include <sphericalsfm/spherical_solvers.h>

namespace sphericalsfm {
    // from Theia library
    int SolveQuartic(const double a, const double b, const double c,
                     const double d, const double e,
                     std::complex<double>* roots) {
      const double a_pw2 = a * a;
      const double b_pw2 = b * b;
      const double a_pw3 = a_pw2 * a;
      const double b_pw3 = b_pw2 * b;
      const double a_pw4 = a_pw3 * a;
      const double b_pw4 = b_pw3 * b;

      const double alpha = -3.0 * b_pw2 / (8.0 * a_pw2) + c / a;
      const double beta =
          b_pw3 / (8.0 * a_pw3) - b * c / (2.0 * a_pw2) + d / a;
      const double gamma =
          -3.0 * b_pw4 / (256.0 * a_pw4) + b_pw2 * c / (16.0 * a_pw3) -
          b * d / (4.0 * a_pw2) + e / a;

      const double alpha_pw2 = alpha * alpha;
      const double alpha_pw3 = alpha_pw2 * alpha;

      const std::complex<double> P(-alpha_pw2 / 12.0 - gamma, 0);
      const std::complex<double> Q(
          -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - std::pow(beta, 2.0) / 8.0,
          0);
      const std::complex<double> R =
          -Q / 2.0 +
          std::sqrt(std::pow(Q, 2.0) / 4.0 + std::pow(P, 3.0) / 27.0);

      const std::complex<double> U = std::pow(R, (1.0 / 3.0));
      std::complex<double> y;

      const double kEpsilon = 1e-8;
      if (std::abs(U.real()) < kEpsilon) {
        y = -5.0 * alpha / 6.0 - std::pow(Q, (1.0 / 3.0));
      } else {
        y = -5.0 * alpha / 6.0 - P / (3.0 * U) + U;
      }

      const std::complex<double> w = std::sqrt(alpha + 2.0 * y);

      roots[0] =
          -b / (4.0 * a) +
          0.5 * (w + std::sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
      roots[1] =
          -b / (4.0 * a) +
          0.5 * (w - std::sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
      roots[2] =
          -b / (4.0 * a) +
          0.5 * (-w + std::sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
      roots[3] =
          -b / (4.0 * a) +
          0.5 * (-w - std::sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));

      return 4;
    }

    // Provides solutions to the equation a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 using
    // Ferrari's method to reduce to problem to a depressed cubic.
    int SolveQuarticReals(const double a, const double b,
                          const double c, const double d,
                          const double e, double* roots) {
      std::complex<double> complex_roots[4];
      int num_complex_solutions = SolveQuartic(a, b, c, d, e, complex_roots);
      int num_real_solutions = 0;
      for (int i = 0; i < num_complex_solutions; i++) {
        roots[num_real_solutions++] = complex_roots[i].real();
      }
      return num_real_solutions;
    }

    int SolveQuarticReals(const double a, const double b,
                          const double c, const double d,
                          const double e, const double tolerance,
                          double* roots) {
      std::complex<double> complex_roots[4];
      int num_complex_solutions = SolveQuartic(a, b, c, d, e, complex_roots);
      int num_real_solutions = 0;
      for (int i = 0; i < num_complex_solutions; i++) {
        if (std::abs(complex_roots[i].imag()) < tolerance) {
          roots[num_real_solutions++] = complex_roots[i].real();
        }
      }
      return num_real_solutions;
    }



    int spherical_solver_action_matrix(const RayPairList &correspondences, const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es)
    {
        const int N = sample.size();
        if ( N < 3 )
        {
            std::cout << "bad sample size: " << N << "\n";
            return 0;
        }
        
        Eigen::MatrixXd A(N,6);

        for ( int i = 0 ; i < N; i++ )
        {
            const int ind = sample[i];
            const Eigen::Vector3d u = correspondences[ind].first;
            const Eigen::Vector3d v = correspondences[ind].second;
        
            A.row(i) << u(0)*v(0) - u(1)*v(1), u(0)*v(1) + u(1)*v(0), u(2)*v(0), u(2)*v(1), u(0)*v(2), u(1)*v(2);
        }
        
        //Eigen::MatrixXd Q = A.jacobiSvd(Eigen::ComputeFullV).matrixV();
        // QR(A.') --> last rows of Q are nullspace
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
        
        Es->clear();
        for ( int i = 0; i < 4; i++ )
        {
            //if ( fabs(evalues(i).imag()) > 1e-12 ) continue;
            
            Eigen::Vector3d bsoln( V(1,i).real(), V(2,i).real(), V(3,i).real() );
            Eigen::Matrix<double,6,1> psoln( B*bsoln );
            
            Eigen::Matrix3d Esoln;
            Esoln <<
            psoln(0), psoln(1), psoln(2),
            psoln(1), -psoln(0), psoln(3),
            psoln(4), psoln(5), 0;
            
            Esoln /= Esoln.norm();
            
            Es->push_back(Esoln);
        }
        
        return Es->size();
    }

    int spherical_solver_polynomial(const RayPairList &correspondences, const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es)
    {
        const int N = sample.size();
        if ( N != 3 )
        {
            std::cout << "bad sample size: " << N << "\n";
            return 0;
        }
        
        Eigen::MatrixXd A(N,6);

        for ( int i = 0 ; i < N; i++ )
        {
            const int ind = sample[i];
            const Eigen::Vector3d u = correspondences[ind].first;
            const Eigen::Vector3d v = correspondences[ind].second;
        
            A.row(i) << u(0)*v(0) - u(1)*v(1), u(0)*v(1) + u(1)*v(0), u(2)*v(0), u(2)*v(1), u(0)*v(2), u(1)*v(2);
        }
        
        //Eigen::MatrixXd Q = A.jacobiSvd(Eigen::ComputeFullV).matrixV();
        // QR(A.') --> last rows of Q are nullspace
        Eigen::MatrixXd Q = A.transpose().colPivHouseholderQr().householderQ();
        Eigen::Matrix<double,6,3> B = Q.block(0,3,6,3);

        const double   t2 = B(0,2)*B(0,2);
        const double   t3 = B(1,2)*B(1,2);
        const double   t4 = B(2,0)*B(2,0);
        const double   t5 = B(2,1)*B(2,1);
        const double   t6 = B(2,2)*B(2,2);
        const double   t7 = B(3,0)*B(3,0);
        const double   t8 = B(3,1)*B(3,1);
        const double   t9 = B(3,2)*B(3,2);
        const double   t10 = B(4,0)*B(4,0);
        const double   t11 = B(4,1)*B(4,1);
        const double   t12 = B(4,2)*B(4,2);
        const double   t13 = B(5,0)*B(5,0);
        const double   t14 = B(5,1)*B(5,1);
        const double   t15 = B(5,2)*B(5,2);
        const double   t16 = B(0,0)*B(2,0);
        const double   t17 = B(0,0)*B(2,1);
        const double   t18 = B(0,1)*B(2,0);
        const double   t19 = B(0,0)*B(2,2);
        const double   t20 = B(0,1)*B(2,1);
        const double   t21 = B(0,2)*B(2,0);
        const double   t22 = B(0,1)*B(2,2);
        const double   t23 = B(0,2)*B(2,1);
        const double   t24 = B(0,2)*B(2,2);
        const double   t25 = B(0,0)*B(3,0);
        const double   t26 = B(1,0)*B(2,0);
        const double   t27 = B(0,1)*B(3,0);
        const double   t28 = B(1,0)*B(2,1);
        const double   t29 = B(1,1)*B(2,0);
        const double   t30 = B(0,0)*B(3,2);
        const double   t31 = B(0,2)*B(3,0);
        const double   t32 = B(1,0)*B(2,2);
        const double   t33 = B(1,1)*B(2,1);
        const double   t34 = B(1,2)*B(2,0);
        const double   t35 = B(0,1)*B(3,2);
        const double   t36 = B(1,1)*B(2,2);
        const double   t37 = B(1,2)*B(2,1);
        const double   t38 = B(0,2)*B(3,2);
        const double   t39 = B(1,2)*B(2,2);
        const double   t40 = B(0,0)*B(4,0);
        const double   t41 = B(1,0)*B(3,0);
        const double   t42 = B(0,1)*B(4,0);
        const double   t43 = B(1,1)*B(3,0);
        const double   t44 = B(2,0)*B(2,1);
        const double   t45 = B(0,0)*B(4,2);
        const double   t46 = B(0,2)*B(4,0);
        const double   t47 = B(1,0)*B(3,2);
        const double   t48 = B(1,2)*B(3,0);
        const double   t49 = B(2,0)*B(2,2);
        const double   t50 = B(0,1)*B(4,2);
        const double   t51 = B(1,1)*B(3,2);
        const double   t52 = B(2,1)*B(2,2);
        const double   t53 = B(0,2)*B(4,2);
        const double   t54 = B(1,2)*B(3,2);
        const double   t55 = B(0,0)*B(5,0);
        const double   t56 = B(1,0)*B(4,0);
        const double   t57 = B(2,0)*B(3,0);
        const double   t58 = B(0,1)*B(5,0);
        const double   t59 = B(1,1)*B(4,0);
        const double   t60 = B(0,0)*B(5,2);
        const double   t61 = B(0,2)*B(5,0);
        const double   t62 = B(1,0)*B(4,2);
        const double   t63 = B(1,2)*B(4,0);
        const double   t64 = B(2,0)*B(3,2);
        const double   t65 = B(2,2)*B(3,0);
        const double   t66 = B(0,1)*B(5,2);
        const double   t67 = B(1,1)*B(4,2);
        const double   t68 = B(0,2)*B(5,2);
        const double   t69 = B(1,2)*B(4,2);
        const double   t70 = B(2,2)*B(3,2);
        const double   t71 = B(1,0)*B(5,0);
        const double   t72 = B(2,0)*B(4,0);
        const double   t73 = B(1,1)*B(5,0);
        const double   t74 = B(3,0)*B(3,1);
        const double   t75 = B(1,0)*B(5,2);
        const double   t76 = B(1,2)*B(5,0);
        const double   t77 = B(2,0)*B(4,2);
        const double   t78 = B(2,2)*B(4,0);
        const double   t79 = B(3,0)*B(3,2);
        const double   t80 = B(1,1)*B(5,2);
        const double   t81 = B(3,1)*B(3,2);
        const double   t82 = B(1,2)*B(5,2);
        const double   t83 = B(2,2)*B(4,2);
        const double   t84 = B(2,0)*B(5,0);
        const double   t85 = B(3,0)*B(4,0);
        const double   t86 = B(2,0)*B(5,2);
        const double   t87 = B(2,2)*B(5,0);
        const double   t88 = B(3,0)*B(4,2);
        const double   t89 = B(3,2)*B(4,0);
        const double   t90 = B(2,2)*B(5,2);
        const double   t91 = B(3,2)*B(4,2);
        const double   t92 = B(3,0)*B(5,0);
        const double   t93 = B(3,0)*B(5,2);
        const double   t94 = B(3,2)*B(5,0);
        const double   t95 = B(4,0)*B(4,2);
        const double   t96 = B(3,2)*B(5,2);
        const double   t97 = B(4,0)*B(5,0);
        const double   t98 = B(4,0)*B(5,2);
        const double   t99 = B(4,2)*B(5,0);
        const double   t100 = B(4,2)*B(5,2);
        const double   t101 = B(5,0)*B(5,2);
        const double   t102 = B(0,0)/2.0;
        const double   t103 = B(0,1)/2.0;
        const double   t104 = B(0,2)/2.0;
        const double   t105 = B(1,0)/2.0;
        const double   t106 = B(1,1)/2.0;
        const double   t107 = B(1,2)/2.0;
        const double   t108 = B(3,0)/2.0;
        const double   t109 = B(3,1)/2.0;
        const double   t110 = B(3,2)/2.0;
        const double   t111 = B(3,2)*(3.0/2.0);
        const double   t112 = B(4,0)/2.0;
        const double   t113 = B(4,2)/2.0;
        const double   t114 = B(4,2)*(3.0/2.0);
        const double   t115 = B(5,0)/2.0;
        const double   t116 = B(5,2)/2.0;
        const double   t117 = B(5,2)*(3.0/2.0);
        const double   t118 = -t25;
        const double   t119 = -t27;
        const double   t120 = -t30;
        const double   t121 = -t31;
        const double   t122 = -t32;
        const double   t123 = -t35;
        const double   t124 = -t38;
        const double   t125 = -t39;
        const double   t126 = -t40;
        const double   t127 = -t45;
        const double   t128 = -t46;
        const double   t129 = -t49;
        const double   t130 = -t53;
        const double   t131 = -t55;
        const double   t132 = -t59;
        const double   t133 = -t60;
        const double   t134 = -t61;
        const double   t135 = -t62;
        const double   t136 = -t67;
        const double   t137 = -t68;
        const double   t138 = -t69;
        const double   t139 = -t75;
        const double   t140 = -t79;
        const double   t141 = -t82;
        const double   t142 = -t83;
        const double   t143 = -t85;
        const double   t144 = -t88;
        const double   t145 = -t89;
        const double   t146 = -t90;
        const double   t147 = -t91;
        const double   t148 = -t92;
        const double   t149 = -t93;
        const double   t150 = -t94;
        const double   t151 = -t96;
        const double   t152 = -t100;
        const double   t153 = -t101;
        const double   t154 = -t102;
        const double   t155 = -t103;
        const double   t156 = -t104;
        const double   t157 = -t105;
        const double   t158 = -t106;
        const double   t159 = -t107;
        const double   t160 = -t108;
        const double   t161 = -t109;
        const double   t162 = -t110;
        const double   t163 = -t112;
        const double   t164 = -t113;
        const double   t165 = -t115;
        const double   t166 = -t116;
        const double   t167 = t4/2.0;
        const double   t168 = t5/2.0;
        const double   t169 = t6/2.0;
        const double   t170 = t7/2.0;
        const double   t171 = t8/2.0;
        const double   t172 = t9/2.0;
        const double   t173 = t9*(3.0/2.0);
        const double   t174 = t10/2.0;
        const double   t175 = t12/2.0;
        const double   t176 = t12*(3.0/2.0);
        const double   t177 = t13/2.0;
        const double   t178 = t15/2.0;
        const double   t179 = t15*(3.0/2.0);
        const double   t180 = t16+t41;
        const double   t181 = t24+t54;
        const double   t182 = t40+t71;
        const double   t183 = t42+t73;
        const double   t184 = t45+t75;
        const double   t185 = t50+t80;
        const double   t186 = t53+t82;
        const double   t187 = t70+t100;
        const double   t188 = t90+t91;
        const double   t189 = t98+t99;
        const double   t190 = t12+t15;
        const double   t209 = t2+t3+t9;
        const double   t213 = t19+t21+t47+t48;
        const double   t191 = -t167;
        const double   t192 = -t169;
        const double   t193 = -t170;
        const double   t194 = -t172;
        const double   t195 = -t175;
        const double   t196 = -t177;
        const double   t197 = -t178;
        const double   t198 = t26+t118;
        const double   t199 = t38+t125;
        const double   t200 = t39+t124;
        const double   t201 = t56+t131;
        const double   t202 = t58+t132;
        const double   t203 = t60+t135;
        const double   t204 = t66+t136;
        const double   t205 = t68+t138;
        const double   t206 = t69+t137;
        const double   t207 = t70+t152;
        const double   t208 = t83+t151;
        const double   t210 = t130+t141;
        const double   t211 = t144+t145;
        const double   t212 = t149+t150;
        const double   t214 = t46+t76+t184;
        const double   t215 = t32+t34+t120+t121;
        const double   t216 = t62+t63+t133+t134;
        const double   t222 = t2+t3+t169+t172+t175+t178;
        const double   t217 = t172+t175+t192+t197;
        const double   t218 = t169+t175+t194+t197;
        const double   t219 = t175+t179+t192+t194;
        const double   t220 = t176+t178+t192+t194;
        const double   t221 = t169+t173+t195+t197;
        Eigen::Matrix<double,6,10> C;
        C(0,0) = B(1,0)*t4*(-1.0/2.0)-(B(1,0)*t13)/2.0+B(3,0)*t16+B(5,0)*t126+t7*t105+t10*t105;
        C(0,1) = B(3,1)*t180-B(5,1)*t182-B(1,1)*(t167-t174+t177+t193)+B(2,1)*(t25-t26)-B(4,1)*(t55-t56)+B(0,1)*(t57-t97);
        C(0,2) = B(1,0)*t5*(-1.0/2.0)-(B(1,0)*t14)/2.0-B(4,1)*t202+t8*t105+t11*t105+B(2,1)*(t27-t29)-B(5,1)*(t183+B(0,0)*B(4,1))+B(3,1)*(t17+t18+t43);
        C(0,3) = B(1,2)*t4*(-1.0/2.0)-(B(1,2)*t13)/2.0-B(4,0)*t203+t7*t107+t10*t107+B(3,0)*(t19+t21+t47)+B(2,0)*(t30+t122)-B(5,0)*(t46+t184);
        C(0,4) = B(3,1)*t213-B(5,1)*t214+B(2,1)*(t30+t31-t34+t122)-B(1,1)*(t49-t95+t101+t140)+B(0,1)*(t64+t65-t189)-B(4,1)*(t61-t63+t203);
        C(0,5) = B(0,0)*t207+B(3,0)*t181+B(2,0)*t199-B(5,0)*t186-B(4,0)*t205-B(1,0)*(t169+t178+t194+t195);
        C(0,6) = B(1,1)*t5*(-1.0/2.0)-(B(1,1)*t14)/2.0+B(3,1)*t20+t8*t106+t11*t106+(-B(0,1)*B(4,1))*B(5,1);
        C(0,7) = B(1,2)*t5*(-1.0/2.0)-(B(1,2)*t14)/2.0-B(4,1)*t204+t8*t107+t11*t107+B(2,1)*(t35-t36)-B(5,1)*(t185+B(0,2)*B(4,1))+B(3,1)*(t22+t23+t51);
        C(0,8) = B(0,1)*t207+B(3,1)*t181+B(2,1)*t199-B(5,1)*t186-B(4,1)*t205-B(1,1)*(t169+t178+t194+t195);
        C(0,9) = B(3,2)*t24+B(1,2)*t209-B(1,2)*t222-B(4,2)*t205;
        C(1,0) = t13*t112+(B(4,0)*B(4,0))*t112-B(4,0)*(t167+t170);
        C(1,1) = -B(2,1)*t72+B(5,1)*t97+B(3,1)*t143+B(4,1)*(t10*(3.0/2.0)+t177+t191+t193);
        C(1,2) = B(4,0)*t5*(-1.0/2.0)-(B(4,0)*t8)/2.0+B(4,0)*t11*(3.0/2.0)+t14*t112+(B(4,1)*B(5,0))*B(5,1)-B(4,1)*(t44+t74);
        C(1,3) = B(4,2)*t4*(-1.0/2.0)-(B(4,2)*t7)/2.0+B(5,0)*t98+t10*t114+t13*t113-B(4,0)*(t49+t79);
        C(1,4) = B(5,1)*t189-B(4,1)*(t49+t79-t95*3.0+t153)-B(2,1)*(t77+t78)-B(3,1)*(t88+t89);
        C(1,5) = B(5,0)*t100+B(2,0)*t142+B(3,0)*t147-B(4,0)*(t169+t172-t176+t197);
        C(1,6) = (B(4,1)*B(4,1)*B(4,1))/2.0+(B(4,1)*t14)/2.0-B(4,1)*(t168+t171);
        C(1,7) = B(4,2)*t5*(-1.0/2.0)-(B(4,2)*t8)/2.0+t11*t114+t14*t113+(B(4,1)*B(5,2))*B(5,1)-B(4,1)*(t52+t81);
        C(1,8) = B(5,1)*t100+B(2,1)*t142+B(3,1)*t147-B(4,1)*(t169+t172-t176+t197);
        C(1,9) = B(0,2)*t186-B(1,2)*t205+B(4,2)*t190-B(4,2)*t222;
        C(2,0) = B(0,0)*t7*(-1.0/2.0)-(B(0,0)*t13)/2.0+B(3,0)*t26+B(5,0)*t56+t4*t102+t10*t102;
        C(2,1) = B(2,1)*t180+B(4,1)*t182-B(3,1)*(t25-t26)-B(5,1)*(t55-t56)+B(0,1)*(t167+t174+t193+t196)+B(1,1)*(t57+t97);
        C(2,2) = B(0,0)*t8*(-1.0/2.0)-(B(0,0)*t14)/2.0+B(4,1)*t183+B(5,1)*(-t58+t59+B(1,0)*B(4,1))+t5*t102+t11*t102+B(3,1)*(t28+t29+t119)+B(2,1)*(t18+t43);
        C(2,3) = B(0,2)*t7*(-1.0/2.0)-(B(0,2)*t13)/2.0+B(4,0)*t184+t4*t104+t10*t104+B(3,0)*(t32+t34+t120)+B(5,0)*(t62+t63+t133)+B(2,0)*(t19+t47);
        C(2,4) = B(2,1)*t213+B(4,1)*t214-B(3,1)*(t30+t31-t34+t122)+B(1,1)*(t64+t65+t189)-B(5,1)*(t61-t63+t203)+B(0,1)*(t49+t95+t140+t153);
        C(2,5) = B(1,0)*t187+B(2,0)*t181+B(0,0)*t218+B(4,0)*t186-B(3,0)*t199-B(5,0)*t205;
        C(2,6) = B(0,1)*t8*(-1.0/2.0)-(B(0,1)*t14)/2.0+B(3,1)*t33+t5*t103+t11*t103+(B(1,1)*B(4,1))*B(5,1);
        C(2,7) = B(0,2)*t8*(-1.0/2.0)-(B(0,2)*t14)/2.0+B(4,1)*t185+B(5,1)*(-t66+t67+B(1,2)*B(4,1))+t5*t104+t11*t104+B(3,1)*(t36+t37+t123)+B(2,1)*(t22+t51);
        C(2,8) = B(1,1)*t187+B(2,1)*t181+B(0,1)*t218+B(4,1)*t186-B(3,1)*t199-B(5,1)*t205;
        C(2,9) = B(3,2)*t39-B(0,2)*t209+B(0,2)*t222-B(5,2)*t205;
        C(3,0) = (B(5,0)*B(5,0))*t115-B(5,0)*(t167+t170-t174);
        C(3,1) = -B(2,1)*t84+B(4,1)*t97+B(3,1)*t148+B(5,1)*(t13*(3.0/2.0)+t174+t191+t193);
        C(3,2) = B(5,0)*t5*(-1.0/2.0)-(B(5,0)*t8)/2.0+B(5,0)*t14*(3.0/2.0)+t11*t115-B(5,1)*(t44+t74-B(4,0)*B(4,1));
        C(3,3) = B(5,2)*t4*(-1.0/2.0)-(B(5,2)*t7)/2.0+t10*t116+t13*t117-B(5,0)*(t49+t79-t95);
        C(3,4) = B(4,1)*t189-B(5,1)*(t49+t79-t95-t101*3.0)-B(2,1)*(t86+t87)-B(3,1)*(t93+t94);
        C(3,5) = B(4,0)*t100+B(2,0)*t146+B(3,0)*t151-B(5,0)*(t169+t172-t179+t195);
        C(3,6) = (B(5,1)*B(5,1)*B(5,1))/2.0-B(5,1)*(t11*(-1.0/2.0)+t168+t171);
        C(3,7) = B(5,2)*t5*(-1.0/2.0)-(B(5,2)*t8)/2.0+t11*t116+t14*t117-B(5,1)*(t52+t81-B(4,1)*B(4,2));
        C(3,8) = B(4,1)*t100+B(2,1)*t146+B(3,1)*t151-B(5,1)*(t169+t172-t179+t195);
        C(3,9) = B(1,2)*t186+B(0,2)*t205+B(5,2)*t190-B(5,2)*t222;
        C(4,0) = B(3,0)*t10*(-1.0/2.0)-(B(3,0)*t13)/2.0+t4*t108+(B(3,0)*B(3,0))*t108;
        C(4,1) = B(2,1)*t57+B(4,1)*t143+B(5,1)*t148+B(3,1)*(t7*(3.0/2.0)+t167-t174+t196);
        C(4,2) = B(3,0)*t8*(3.0/2.0)-(B(3,0)*t11)/2.0-(B(3,0)*t14)/2.0+B(3,1)*t44+t5*t108+(-B(3,1)*B(4,0))*B(4,1)+(-B(3,1)*B(5,0))*B(5,1);
        C(4,3) = B(3,2)*t10*(-1.0/2.0)-(B(3,2)*t13)/2.0+B(3,0)*t49+B(4,0)*t144+B(5,0)*t149+t4*t110+t7*t111;
        C(4,4) = B(3,1)*(t49+t79*3.0-t95+t153)+B(2,1)*(t64+t65)-B(4,1)*(t88+t89)-B(5,1)*(t93+t94);
        C(4,5) = B(2,0)*t70+B(4,0)*t147+B(5,0)*t151+B(3,0)*t221;
        C(4,6) = B(3,1)*t11*(-1.0/2.0)-(B(3,1)*t14)/2.0+t5*t109+(B(3,1)*B(3,1))*t109;
        C(4,7) = B(3,2)*t11*(-1.0/2.0)-(B(3,2)*t14)/2.0+B(3,1)*t52+t5*t110+t8*t111+(-B(3,1)*B(4,2))*B(4,1)+(-B(3,1)*B(5,2))*B(5,1);
        C(4,8) = B(2,1)*t70+B(4,1)*t147+B(5,1)*t151+B(3,1)*t221;
        C(4,9) = B(3,2)*t6+B(3,2)*t209-B(3,2)*t222;
        C(5,0) = B(4,0)*t180-B(5,0)*(t25-t26);
        C(5,1) = B(2,1)*t182+B(4,1)*t180-B(5,1)*(t25-t26)-B(3,1)*(t55-t56)+B(1,1)*(t84+t85)+B(0,1)*(t72+t148);
        C(5,2) = B(2,1)*t183-B(3,1)*t202+B(4,1)*(t17+t18+t43+B(1,0)*B(3,1))-B(5,1)*(t27-t28-t29+B(0,0)*B(3,1));
        C(5,3) = B(2,0)*t184-B(3,0)*t203+B(4,0)*t213-B(5,0)*(t30+t31-t34+t122);
        C(5,4) = B(2,1)*t214+B(4,1)*t213-B(5,1)*(t30+t31-t34+t122)+B(0,1)*(t77+t78+t212)-B(3,1)*(t61-t63+t203)+B(1,1)*(t86+t87+t88+t89);
        C(5,5) = B(1,0)*t188+B(2,0)*t186+B(0,0)*t208+B(4,0)*t181-B(3,0)*t205-B(5,0)*t199;
        C(5,6) = B(4,1)*(t20+B(1,1)*B(3,1))+B(5,1)*(t33-B(0,1)*B(3,1));
        C(5,7) = B(2,1)*t185-B(3,1)*t204+B(4,1)*(t22+t23+t51+B(1,2)*B(3,1))-B(5,1)*(t35-t36-t37+B(0,2)*B(3,1));
        C(5,8) = B(1,1)*t188+B(2,1)*t186+B(0,1)*t208+B(4,1)*t181-B(3,1)*t205-B(5,1)*t199;
        C(5,9) = B(2,2)*t186-B(3,2)*t205;
        
        Eigen::Matrix<double,6,4> G( C.block(0,0,6,6).lu().solve(C.block(0,6,6,4)) );

        const double a = -G(5,0);
        const double b = G(4,0)-G(5,1);
        const double c = G(4,1)-G(5,2);
        const double d = G(4,2)-G(5,3);
        const double e = G(4,3);

        double ysols[4];
        //int nsols = SolveQuarticReals(a,b,c,d,e,1e-15,ysols);
        int nsols = SolveQuarticReals(a,b,c,d,e,ysols);
        
        double xsols[4];
        for ( int i = 0; i < nsols; i++ )
        {
            const double y = ysols[i];
            const double y2 = y*y;
            const double y3 = y2*y;
            xsols[i] = - G(5,0)*y3 - G(5,1)*y2 - G(5,2)*y - G(5,3);
        }

        Es->clear();
        for ( int i = 0; i < nsols; i++ )
        {
            Eigen::Vector3d bsoln( xsols[i], ysols[i], 1 );
            Eigen::Matrix<double,6,1> psoln( B*bsoln );
            
            Eigen::Matrix3d Esoln;
            Esoln <<
            psoln(0), psoln(1), psoln(2),
            psoln(1), -psoln(0), psoln(3),
            psoln(4), psoln(5), 0;
            
            Esoln /= Esoln.norm();
            
            Es->push_back(Esoln);
        }
        
        return Es->size();
    }
    
}
