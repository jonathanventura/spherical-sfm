#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gflags/gflags.h>

#include <problem_generator/problem_generator.h>
#include <sphericalsfm/spherical_estimator.h>
#include "nister_estimator.h"
#include "stewenius_estimator.h"

#include <sphericalsfm/so3.h>

DEFINE_int32(ntrials, 100, "Number of trials");
DEFINE_double(point_noise, 0, "Std. dev. of Gaussian noise added to point observations (px)");
DEFINE_double(focal, 600, "Focal length");
DEFINE_double(rotation, -1, "Rotation amount [deg]");
DEFINE_bool(inward, false, "Inward-facing");
DEFINE_bool(disambiguate, false, "Use extra point to choose best solution");
DEFINE_string(output_path, "out.tab", "Path for output file");
DEFINE_string(timings_path, "timings.tab", "Path for timings file");

using namespace problem_generator;
using namespace sphericalsfm;

void evaluate( EssentialEstimator &estimator, int sample_size, const RelativePoseProblem &prob, FILE *outfile, FILE *timingsfile )
{
    auto start = std::chrono::steady_clock::now();
    std::vector<Eigen::Matrix3d> Es;
    std::vector<int> sample(sample_size);
    for ( int i = 0; i < sample_size; i++ ) sample[i] = i;
    estimator.MinimalSolver( sample, &Es );
    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    double frob_error = INFINITY;
    double rot_error = INFINITY;
    double trans_error = INFINITY;
    if ( FLAGS_disambiguate )
    {
        // choose E
        double best_sampson = INFINITY;
        Eigen::Matrix3d E;
        for ( int i = 0; i < Es.size(); i++ ) 
        {
            double sampson = 0;
            for ( int j = 0; j < prob.correspondences.size(); j++ )
            {
                sampson += estimator.EvaluateModelOnPoint(Es[i],j);
            }
            if ( sampson < best_sampson )   
            {
                best_sampson = sampson;
                E = Es[i];
            }
        }
        frob_error = prob.soln.calc_frob_error(E);
         
        // choose R,t
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        estimator.Decompose(E,&R,&t);
        rot_error = prob.soln.calc_rot_error(R);

        trans_error = prob.soln.calc_trans_error(t);
    } else {
        for ( int i = 0; i < Es.size(); i++ ) 
        {
            double my_frob_err = prob.soln.calc_frob_error(Es[i]);
            if ( my_frob_err < frob_error ) frob_error = my_frob_err;
            
            Eigen::Matrix3d R1, R2;
            Eigen::Vector3d t;
            DecomposeEssentialMatrix(Es[i], &R1, &R2, &t );
            
            double my_rot_err = std::min(prob.soln.calc_rot_error(R1),prob.soln.calc_rot_error(R2));
            if ( my_rot_err < rot_error ) rot_error = my_rot_err;

            double my_trans_err = prob.soln.calc_trans_error(t);
            if ( my_trans_err < trans_error ) trans_error = my_trans_err;
        }
    }
    
    fprintf( outfile, "%.15lf\t%.15lf\t%.15lf\t",frob_error,rot_error,trans_error);
    fprintf( timingsfile, "%.15lf\t",time);
}

int main( int argc, char **argv )
{
    srand(1234);
    
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FILE *outfile = fopen(FLAGS_output_path.c_str(),"w");
    FILE *timingsfile = fopen(FLAGS_timings_path.c_str(),"w");
    
    if ( !outfile ) {
        std::cerr << "could not open " << FLAGS_output_path << "\n";
        exit(1);
    }
    
    ProblemGenerator generator(FLAGS_point_noise/FLAGS_focal);

    for ( int i = 0; i < FLAGS_ntrials; i++ )
    {
        RelativePoseProblem prob = generator.make_random_problem(6,FLAGS_inward,FLAGS_rotation*M_PI/180);
        
        bool non_minimal = ( FLAGS_point_noise>0 );
        SphericalEstimator spherical_eig_estimator(prob.correspondences,false,FLAGS_inward);
        SphericalEstimator spherical_poly_estimator(prob.correspondences,true,FLAGS_inward);
        NisterEstimator nister_estimator(prob.correspondences);
        SteweniusEstimator stewenius_estimator(prob.correspondences);

        evaluate(spherical_eig_estimator,(non_minimal?5:3),prob,outfile,timingsfile);
        evaluate(spherical_poly_estimator,(non_minimal?5:3),prob,outfile,timingsfile);
        evaluate(nister_estimator,5,prob,outfile,timingsfile);
        evaluate(stewenius_estimator,5,prob,outfile,timingsfile);
         
        fprintf( outfile, "\n" );
        fprintf( timingsfile, "\n" );
    }
    fclose(outfile);
    fclose(timingsfile);

    return 0;
}
