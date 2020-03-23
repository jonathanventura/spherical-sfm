
#include <sphericalsfm/so3.h>

namespace sphericalsfm {

    Eigen::Matrix3d skew3(const Eigen::Vector3d &v )
    {
        Eigen::Matrix3d s;
        s <<
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
        return s;
    }
        
    Eigen::Matrix3d so3exp(const Eigen::Vector3d &r)
    {
        if ( r.norm() < 1e-10 ) return Eigen::Matrix3d::Identity();
        const double theta = r.norm();
        const Eigen::Matrix3d K = skew3(r/theta);
        const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + sin(theta)*K+(1-cos(theta))*K*K;
        return R;
    }

    Eigen::Vector3d so3ln(const Eigen::Matrix3d &R)
    {
         Eigen::Vector3d result;
         
         const double cos_angle = (R(0,0) + R(1,1) + R(2,2) - 1.0) * 0.5;
         result[0] = (R(2,1)-R(1,2))/2;
         result[1] = (R(0,2)-R(2,0))/2;
         result[2] = (R(1,0)-R(0,1))/2;
         
         double sin_angle_abs = sqrt(result.dot(result));
         if (cos_angle > M_SQRT1_2) {            // [0 - Pi/4[ use asin
             if(sin_angle_abs > 0){
                 result *= asin(sin_angle_abs) / sin_angle_abs;
             }
         } else if( cos_angle > -M_SQRT1_2) {    // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
             double angle = acos(cos_angle);
             result *= angle / sin_angle_abs;
         } else {  // rest use symmetric part
             // antisymmetric part vanishes, but still large rotation, need information from symmetric part
             const double angle = M_PI - asin(sin_angle_abs);
             const double d0 = R(0,0) - cos_angle,
                 d1 = R(1,1) - cos_angle,
                 d2 = R(2,2) - cos_angle;
             Eigen::Vector3d r2;
             if(fabs(d0) > fabs(d1) && fabs(d0) > fabs(d2)){ // first is largest, fill with first column
                 r2[0] = d0;
                 r2[1] = (R(1,0)+R(0,1))/2;
                 r2[2] = (R(0,2)+R(2,0))/2;
             } else if(fabs(d1) > fabs(d2)) {                // second is largest, fill with second column
                 r2[0] = (R(1,0)+R(0,1))/2;
                 r2[1] = d1;
                 r2[2] = (R(2,1)+R(1,2))/2;
             } else {                                // third is largest, fill with third column
                 r2[0] = (R(0,2)+R(2,0))/2;
                 r2[1] = (R(2,1)+R(1,2))/2;
                 r2[2] = d2;
             }
             // flip, if we point in the wrong direction!
             if(r2.dot(result) < 0)
                 r2 *= -1;
             r2 = r2/r2.norm();
             result = angle * r2;
         }
        return result;
    }
}
