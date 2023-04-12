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
#include <five_point/nister_estimator.h>
#include <five_point/stewenius_estimator.h>

#include <RansacLib/ransac.h>

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

typedef ransac_lib::LocallyOptimizedMSAC<Eigen::Matrix3d,std::vector<Eigen::Matrix3d>,EssentialEstimator> RansacType;

void evaluate( EssentialEstimator &estimator, const RelativePoseProblem &prob, FILE *outfile, FILE *timingsfile )
{
    auto start = std::chrono::steady_clock::now();

    RansacType ransac;

    ransac_lib::LORansacOptions options;
    options.squared_inlier_threshold_ = pow(2/FLAGS_focal,2);
    options.num_lsq_iterations_ = 0;
    options.num_lo_steps_ = 0;
    options.min_num_iterations_ = 0;
    options.max_num_iterations_ = INT_MAX;
    options.final_least_squares_ = false;

    std::vector<bool> inliers;
    ransac_lib::RansacStatistics ransac_stats;
    Eigen::Matrix3d E;
    int ninliers = ransac.EstimateModel(options, estimator, &E, &ransac_stats);
    int iter = ransac_stats.num_iterations;

    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    estimator.Decompose(E,ransac_stats.inlier_indices,&R,&t);

    double frob_error = prob.soln.calc_frob_error(E);
    double rot_error = prob.soln.calc_rot_error(R);
    double trans_error = prob.soln.calc_trans_error(t);
    
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
        RelativePoseProblem prob = generator.make_random_problem(100,FLAGS_inward,FLAGS_rotation);
        
        SphericalEstimator spherical_eig_estimator(prob.correspondences,false,FLAGS_inward);
        SphericalEstimator spherical_poly_estimator(prob.correspondences,true,FLAGS_inward);
        NisterEstimator nister_estimator(prob.correspondences);
        SteweniusEstimator stewenius_estimator(prob.correspondences);

        evaluate(spherical_eig_estimator,prob,outfile,timingsfile);
        evaluate(spherical_poly_estimator,prob,outfile,timingsfile);
        evaluate(nister_estimator,prob,outfile,timingsfile);
        evaluate(stewenius_estimator,prob,outfile,timingsfile);
         
        fprintf( outfile, "\n" );
        fprintf( timingsfile, "\n" );
    }
    fclose(outfile);
    fclose(timingsfile);

    return 0;
}
