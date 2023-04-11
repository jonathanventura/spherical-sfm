
#include <Eigen/Geometry>

#include <sphericalsfm/so3.h>
#include <problem_generator/random.h>
#include <problem_generator/problem_generator.h>

#include <iostream>

using namespace sphericalsfm;

namespace problem_generator
{
    RelativePoseProblem ProblemGenerator::make_random_problem( int num_corr, bool inward, double rotation )
    {
        RandomGenerator generator;
        
        RelativePoseProblem prob;
        prob.correspondences.resize(num_corr);

        Eigen::Vector3d t1(0,0,1);
        Eigen::Vector3d t2(0,0,1);
        if ( !inward ) { t1[2] = -1; t2[2] = -1; }

        while ( true )
        {
            // sample two random rotations
            Eigen::Matrix3d R1, R2;
            Eigen::Vector3d r1 = generator.rand_unit_vector() * (generator.rand() * M_PI);
            R1 = so3exp(r1);
            
            if ( rotation < 0 )
            {
                Eigen::Vector3d r2 = generator.rand_unit_vector() * (generator.rand() * M_PI);
                R2 = so3exp(r2);
            } else {
                Eigen::Vector3d r = generator.rand_unit_vector() * rotation;
                Eigen::Matrix3d R = so3exp(r);
                R2 = R * R1;
            }
        
            // calculate relative pose
            // x1 = R1*X+t1
            // x2 = R2*X+t2
            // X = R1'*x1-R1.'*t1
            // x2 = R2*(R1.'*x1-R1.'*t1)+t2
            //    = (R2*R1.')*x1+(t2-R2*R1.'*t1)
            Eigen::Matrix3d R = R2 * R1.transpose();
            Eigen::Vector3d t = t2 - R2 * R1.transpose() * t1;
        
            prob.soln.E = skew3(t)*R;
            prob.soln.R = R;
            prob.soln.t = t;
        
            bool good = true;
        
            // generate observations
            for ( int i = 0; i < num_corr; i++ ) {
                RayPair corr;
                corr.first << 0,0,1;
                corr.second << 0,0,1;
                
                // make point in first image
                corr.first.head(2) = generator.rand2();
                Eigen::Vector3d X = corr.first;
                if ( inward ) X *= generator.rand()*0.25 + 0.5;
                else X *= generator.rand()*2 + 6;

                // project to second camera
                Eigen::Vector3d P2X = R * X + t; // transform point to second camera's frame
                if ( P2X(2) < 0 ) good = false;
                corr.second.head(2) = P2X.head(2)/P2X(2); // project to image plane
                
                // add noise to point observations
                corr.first.head(2) += point_noise * generator.randn2();
                corr.second.head(2) += point_noise * generator.randn2();
                
                // store correspondence
                prob.correspondences[i] = corr;
            }
            
            if ( good ) break;
        }
        
        return prob;
    }
}


