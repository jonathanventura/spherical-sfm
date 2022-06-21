#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <sphericalsfm/pose_estimator.h>
#include <RansacLib/ransac.h>

using namespace sphericalsfm;

int main(int argc, char **argv)
{
/*
    FILE *f = fopen("matches.dat","r");
    RayPairList list;
    while ( true )
    {
        Eigen::Vector3d point0;
        Eigen::Vector3d point1;
        if ( fread(point0.data(),sizeof(double),3,f) != 3 ) break;
        if ( fread(point1.data(),sizeof(double),3,f) != 3 ) break;
        if ( point0.norm() > 2000 ) continue;
        if ( point1.norm() > 2000 ) continue;
        std::cout << point0.transpose() << " -> " << point1.transpose() << "\n";
        list.push_back(std::make_pair(point0,point1));
    }
    fclose(f);
    std::cout << "read " << list.size() << " pairs\n";
        
    FILE *pointsf = fopen("points0.obj","w");
    for ( int i = 0; i < list.size(); i++ )
    {
        fprintf(pointsf,"v %.15f %.15f %.15f\n",list[i].first(0),list[i].first(1),list[i].first(2));
    }
    fclose(pointsf);

    pointsf = fopen("points1.obj","w");
    for ( int i = 0; i < list.size(); i++ )
    {
        fprintf(pointsf,"v %.15f %.15f %.15f\n",list[i].second(0),list[i].second(1),list[i].second(2));
    }
    fclose(pointsf);
*/
    const double focal = 500.;
    const double inlier_threshold = 2./focal;
    RayPairList list;
    FILE *f = fopen("matches.dat","r");
    while ( true )
    {
        Eigen::Vector3d point;
        Eigen::Vector3d obs;
        if ( fread(point.data(),sizeof(double),3,f) != 3 ) break;
        if ( fread(obs.data(),sizeof(double),3,f) != 3 ) break;
        if ( point.norm() > 2000 ) continue;
        std::cout << point.transpose() << " -> " << obs.transpose() << "\n";
        list.push_back(std::make_pair(point,obs));
    }
    fclose(f);
    std::cout << "read " << list.size() << " pairs\n";
        
    FILE *pointsf = fopen("points.obj","w");
    for ( int i = 0; i < list.size(); i++ )
    {
        fprintf(pointsf,"v %.15f %.15f %.15f\n",list[i].first(0),list[i].first(1),list[i].first(2));
    }
    fclose(pointsf);
    
    ransac_lib::LORansacOptions options;
    options.squared_inlier_threshold_ = inlier_threshold*inlier_threshold;
    options.final_least_squares_ = true;
    ransac_lib::RansacStatistics stats;

    PoseEstimator estimator( list );
            
    std::cout << "running ransac on " << list.size() << " matches\n";
    ransac_lib::LocallyOptimizedMSAC<Pose,std::vector<Pose>,PoseEstimator> ransac;

    Pose pose;
    int ninliers = ransac.EstimateModel( options, estimator, &pose, &stats );
    std::cout << ninliers << " / " << list.size() << " inliers\n";
    
    std::cout << "r: " << pose.r.transpose() << "\n";
    std::cout << "t: " << pose.t.transpose() << "\n";
}

