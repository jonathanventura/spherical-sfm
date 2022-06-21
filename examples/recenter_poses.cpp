#include <sphericalsfm/so3.h>
#include <sphericalsfm/sfm_types.h>
#include <sphericalsfm/sfm.h>
#include <sphericalsfm/plane_estimator.h>
#include <sphericalsfm/preemptive_ransac.h>

using namespace sphericalsfm;

DEFINE_string(input, "", "Path to input poses file");
DEFINE_string(output, "", "Path to output poses file");

void read_poses( const std::string & posespath, SfM &sfm, std::vector<int> &indices )
{
    FILE *posesf = fopen(posespath.c_str(),"r");
    while ( true )
    {
        int index;
        double t[3];
        double r[3];
        
        int nread = fscanf(posesf,"%d %lf %lf %lf %lf %lf %lf\n",
                           &index,
                           t+0,t+1,t+2,
                           r+0,r+1,r+2);
        if ( nread != 7 ) break;
        
        Pose pose( Eigen::Vector3d(t[0],t[1],t[2]), Eigen::Vector3d(r[0],r[1],r[2]) );
        
        sfm.AddCamera(pose);
        indices.push_back( index );
    }
    fclose(posesf);
}

void recenter( SfM &sfm )
{
    // calculate centroid 
    // shift so that centroid is at origin
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for ( int i = 0; i < sfm.GetNumCameras(); i++ )
    {
        Pose pose = sfm.GetPose(i);
        centroid += pose.getCenter();
    }
    centroid /= sfm.GetNumCameras();

    std::cout << "centroid over " << sfm.GetNumCameras() << " cameras: " << centroid.transpose() << "\n";
    Pose T( -centroid, Eigen::Vector3d::Zero() );
    sfm.Apply(T);
}

void rescale( SfM &sfm )
{
    double avg_scale = 0;
    for ( int i = 0; i < sfm.GetNumCameras(); i++ )
    {
        Pose pose = sfm.GetPose(i);
        avg_scale += pose.getCenter().norm();
    }
    avg_scale /= sfm.GetNumCameras();

    std::cout << "average scale over " << sfm.GetNumCameras() << " cameras: " << avg_scale << "\n";
    sfm.Apply(1./avg_scale);
}

/*
void estimate_plane( SfM &sfm )
{
    RayPairList centers(sfm.GetNumCameras());
    for ( int i = 0; i < sfm.GetNumCameras(); i++ )
    {
        Pose pose = sfm.GetPose(i);
        
        // get camera center
        Eigen::Vector3d c = pose.getCenter();
        Ray ray;
        ray.head(3) = c;
        centers[i] = std::make_pair( ray, ray );
    }
    
    std::vector<PlaneEstimator*> estimators(200);
    for ( int i = 0; i < estimators.size(); i++ ) estimators[i] = new PlaneEstimator;
    
    PreemptiveRANSAC<RayPairList, PlaneEstimator> plane_ransac;
    plane_ransac.inlier_threshold = 0.01;
    
    std::vector<bool> inliers;
    PlaneEstimator *best_estimator = NULL;
    int ninliers = plane_ransac.compute( centers.begin(), centers.end(), estimators, &best_estimator, inliers );
    std::cout << ninliers << "/" << centers.size() << " inliers\n";
    std::cout << best_estimator->normal << "\n";
    Eigen::Vector3d up = best_estimator->normal;
    if ( up(1) < 0 ) up = -up;
    
    // calculate rotation to correct up vector
    Eigen::Matrix3d correction = get_rotation( up, Eigen::Vector3d(0,1,0) );
    std::cout << "correction rotation:\n" << correction << "\n";
    std::cout << "corrected up vector:\n" << correction * up << "\n";
    
    // c = -R.' * t
    // newup = correction * up
    // newc = correction * c = - correction * R.' * t = - (R * correction.') .' * t
    
    Pose T( Eigen::Vector3d::Zero(), -so3ln(correction) );

    // rotate all cameras to correct up vector
    for ( int i = 0; i < sfm.GetNumCameras(); i++ )
    {
        Pose pose = sfm.GetCamera(i);
        pose.PostMultiply(T);
        sfm.SetCamera(i,pose);
    }

    // check if we should flip upside-down
    int nflip = 0;
    for ( int i = 0; i < sfm.GetNumCameras(); i++ )
    {
        Pose pose = sfm.GetCamera(i);
        Keyframe &kf = keyframes[i];
        if ( kf.R(1,1) < 0 ) nflip++;
    }
    
    if ( nflip > keyframes.size() / 2 )
    {
        std::cout << "FLIP\n";
        Eigen::Matrix3d Rflip = Eigen::Matrix3d::Identity();
        Rflip(1,1) = -1;
        Rflip(2,2) = -1;
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            Keyframe &kf = keyframes[i];
            kf.R = kf.R * Rflip;
            kf.r = so3ln(kf.R);
        }
    }
}
*/
int main( int argc, char **argv )
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Intrinsics intrinsics(1,0,0); // doesn't matter
    SfM sfm( intrinsics );
    std::vector<int> indices;

    read_poses( FLAGS_input, sfm, indices );
    recenter( sfm );
    rescale( sfm );
    
    sfm.WriteCameraCentersOBJ( "cameras.obj" );
    sfm.WritePoses( FLAGS_output, indices );
}
