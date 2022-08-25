
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

#include <Eigen/Jacobi>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <iostream>

#include <opencv2/core/utility.hpp>

#include <sphericalsfm/sfm.h>
#include <sphericalsfm/so3.h>

#include <RansacLib/ransac.h>
#include <sphericalsfm/triangulation_estimator.h>

#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>


namespace sphericalsfm {

    struct ReprojectionError
    {
        ReprojectionError( double _focal, double _x, double _y )
        : focal(_focal), x(_x), y(_y)
        {
            
        }
        
        template <typename T>
        bool operator()(const T* const camera_t,
                        const T* const camera_r,
                        const T* const point,
                        T* residuals) const
        {
            // transform from world to camera
            T p[3];
            ceres::AngleAxisRotatePoint(camera_r, point, p);
            p[0] += camera_t[0]; p[1] += camera_t[1]; p[2] += camera_t[2];
            
            // projection
            T xp = p[0] / p[2];
            T yp = p[1] / p[2];
            
            // intrinsics
            T fxp = T(focal) * xp;
            T fyp = T(focal) * yp;
            
            // residuals
            residuals[0] = fxp - T(x);
            residuals[1] = fyp - T(y);
            
            return true;
        }
        
        double focal, x, y;
    };
    
    template <typename T>
    Eigen::Matrix<T,3,3> skew3(const Eigen::Matrix<T,3,1> &v )
    {
        Eigen::Matrix<T,3,3> s;
        s <<
        T(0), -v[2], v[1],
        v[2], T(0), -v[0],
        -v[1], v[0], T(0);
        return s;
    }
    
    SfM::SfM( const Intrinsics &_intrinsics )
    : intrinsics( _intrinsics ),
    numCameras( 0 ),
    numPoints( 0 ),
    nextCamera( -1 ),
    nextPoint( 0 )
    {
    }

    double * SfM::GetCameraPtr( int camera )
    {
        return (double*)&cameras(camera);
    }

    double * SfM::GetPointPtr( int point )
    {
        return (double*)&points(point);
    }

    int SfM::AddCamera( const Pose &initial_pose, const std::string &path )
    {
        nextCamera++;
        numCameras++;
        
        cameras( nextCamera ).head(3) = initial_pose.t;
        cameras( nextCamera ).tail(3) = initial_pose.r;
        paths( nextCamera ) = path;
        rotationFixed( nextCamera ) = false;
        translationFixed( nextCamera ) = false;
        
        return nextCamera;
    }

    int SfM::AddPoint( const Point &initial_position, const cv::Mat &descriptor )
    {
        numPoints++;
        
        points( nextPoint ) = initial_position;
        pointFixed( nextPoint ) = false;
        
        cv::Mat descriptor_copy;
        descriptor.copyTo( descriptor_copy );
        descriptors( nextPoint ) = descriptor_copy;

        return nextPoint++;
    }

    void SfM::MergePoint( int point1, int point2 )
    {
        for ( int i = 0; i < numCameras; i++ )
        {
            if ( !cameras.exists(i) ) continue;
            if ( !( observations.exists(i,point2) ) ) continue;
            
            observations(i,point1) = observations(i,point2);
        }
        
        RemovePoint( point2 );
    }

    void SfM::AddObservation( int camera, int point, const Observation &observation )
    {
        observations(camera,point) = observation;
    }

    bool SfM::GetObservation( int camera, int point, Observation &observation )
    {
        if ( !observations.exists(camera,point) ) return false;

        observation = observations(camera,point);
        
        return true;
    }

    void SfM::Retriangulate()
    {
        cv::parallel_for_(cv::Range(0,numPoints), [&](const cv::Range &range){
        for ( int j = range.start; j < range.end; j++ )
        //for ( int j = 0; j < numPoints; j++ )
        {
            if ( !points.exists(j) ) continue;
            
            std::vector<TriangulationObservation> tri_observations;
            
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !cameras.exists(i) ) continue;
                if ( !( observations.exists(i,j) ) ) continue;
                tri_observations.push_back( TriangulationObservation( GetPose(i), observations(i,j), intrinsics.focal )) ;
            }

            SetPoint( j, Eigen::Vector3d::Zero() );
            if ( tri_observations.size() < 3 ) continue;
            
            ransac_lib::LORansacOptions options;
            options.squared_inlier_threshold_ = 4.*(intrinsics.focal*intrinsics.focal);
            options.final_least_squares_ = true;
            ransac_lib::RansacStatistics stats;

            TriangulationEstimator estimator( tri_observations );
            
            ransac_lib::LocallyOptimizedMSAC<Point,std::vector<Point>,TriangulationEstimator> ransac;
            
            Point X;
            int ninliers = ransac.EstimateModel( options, estimator, &X, &stats );
            if ( ninliers < 3 ) continue;
            
            SetPoint( j, X );
        }
        });
    }

    void SfM::PreOptimize()
    {
        loss_function = new ceres::CauchyLoss( 1.0 );
        //loss_function = NULL;
    }

    void SfM::ConfigureSolverOptions( ceres::Solver::Options &options, ceres::LinearSolverType linear_solver_type )
    {
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = linear_solver_type;//ceres::SPARSE_SCHUR;
        //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        //options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 2000;
        options.max_num_consecutive_invalid_steps = 100;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 16;
        //options.update_state_every_iteration = true;
        //options.callbacks.push_back(new ScaleCallback(this));
    }

    void SfM::AddResidual( ceres::Problem &problem, int camera, int point )
    {
        Observation vec = observations(camera,point);
        
        ReprojectionError *reproj_error = new ReprojectionError(intrinsics.focal,vec(0),vec(1));
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(reproj_error);
        problem.AddResidualBlock(cost_function, loss_function, GetCameraPtr(camera), GetCameraPtr(camera)+3, GetPointPtr(point) );

        if ( translationFixed( camera ) ) problem.SetParameterBlockConstant( GetCameraPtr(camera) );
        if ( rotationFixed( camera ) ) problem.SetParameterBlockConstant( GetCameraPtr(camera)+3 );
        if ( pointFixed( point ) ) problem.SetParameterBlockConstant( GetPointPtr(point) );
    }

    bool SfM::Optimize()
    {
        if ( numCameras == 0 || numPoints == 0 ) return false;
        
        ceres::Problem problem;
        
        PreOptimize();
        
        bool added_one_camera = false;
        
        std::cout << "\tBuilding BA problem...\n";

        for ( int j = 0; j < numPoints; j++ )
        {
            if ( !points.exists(j) ) continue;
            if ( points(j).norm() == 0 ) continue;
            
            int nobs = 0;
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !cameras.exists(i) ) continue;
                if ( !( observations.exists(i,j) ) ) continue;
                
                nobs++;
            }
            
            if ( nobs < 3 ) continue;
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !cameras.exists(i) ) continue;
                if ( !( observations.exists(i,j) ) ) continue;

                AddResidual( problem, i, j );
                added_one_camera = true;
            }
        }
            
        if ( !added_one_camera ) {
            std::cout << "didn't add any cameras\n";
            return false;
        }
        
        std::cout << "Running optimizer...\n";
        std::cout << "\t" << problem.NumResiduals() << " residuals\n";

        ceres::Solver::Options options;
        ConfigureSolverOptions( options, ceres::SPARSE_SCHUR );
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        if ( summary.termination_type == ceres::FAILURE )
        {
            std::cout << "error: ceres failed.\n";
            exit(1);
        }

        //ScaleCallback callback(this);
        //callback.rescale();
        
        PostOptimize();
        
        return ( summary.termination_type == ceres::CONVERGENCE );
    }

    void SfM::PostOptimize()
    {
        
    }
    
    void SfM::Apply( const Pose &pose )
    {
        // x = PX
        // X -> pose*X
        // x = P' * (pose*X)
        // x = (P*poseinv) * (pose*X)
        Pose poseinv = pose.inverse();
        for ( int i = 0; i < numCameras; i++ )
        {
            Pose campose = GetPose(i);
            campose.postMultiply( poseinv );
            SetPose( i, campose );
        }
        for ( int j = 0; j < numPoints; j++ )
        {
            if ( !points.exists(j) ) continue;
            Point X = GetPoint(j);
            if ( X.norm() == 0 ) continue;
            X = pose.apply(X);
            SetPoint( j, X );
        }
    }

    void SfM::Apply( double scale )
    {
        for ( int i = 0; i < numCameras; i++ )
        {
            Pose campose = GetPose(i);
            campose.t *= scale;
            campose.P.block<3,1>(0,3) = campose.t;
            SetPose( i, campose );
        }
        for ( int j = 0; j < numPoints; j++ )
        {
            if ( !points.exists(j) ) continue;
            Point X = GetPoint(j);
            if ( X.norm() == 0 ) continue;
            X *= scale;
            SetPoint( j, X );
        }
    }

    void SfM::Unapply( const Pose &pose )
    {
        // x = PX
        // X -> poseinv*X
        // x = P' * (poseinv*X)
        // x = (P*pose) * (poseinv*X)
        for ( int i = 0; i < numCameras; i++ )
        {
            Pose campose = GetPose(i);
            campose.postMultiply( pose );
            SetPose( i, campose );
        }
        Pose poseinv = pose.inverse();
        for ( int j = 0; j < numPoints; j++ )
        {
            Point X = GetPoint(j);
            X = poseinv.apply(X);
            SetPoint( j, X );
        }
    }

    Pose SfM::GetPose( int camera )
    {
        if ( !cameras.exists(camera) ) return Pose();
        
        return Pose( cameras( camera ).head(3), cameras( camera ).tail(3) );
    }

    void SfM::SetPose( int camera, const Pose &pose )
    {
        cameras( camera ).head(3) = pose.t;
        cameras( camera ).tail(3) = pose.r;
    }

    Point SfM::GetPoint( int point )
    {
        if ( !points.exists( point ) ) return Point(0,0,0);
        return points( point );
    }

    void SfM::SetPoint( int point, const Point &position )
    {
        points( point ) = position;
    }

    void SfM::RemovePoint( int point )
    {
        for ( int i = 0; i < numCameras; i++ )
        {
            observations.erase(i,point);
        }
        points.erase( point );
        descriptors.erase( point );
    }

    void SfM::RemoveCamera( int camera )
    {
        cameras.erase(camera);
        observations.erase(camera);
        
        for ( int j = 0; j < numPoints; j++ )
        {
            int i = 0;
            for ( ; i < numCameras; i++ )
            {
                if ( observations.exists(i,j) ) break;
            }
            if ( i == numCameras ) points.erase(j);
        }
    }

    void SfM::WritePoses( const std::string &path, const std::vector<int> &indices )
    {
        assert(indices.size() == numCameras);
        FILE *f = fopen( path.c_str(), "w" );
        
        for ( int i = 0; i < numCameras; i++ )
        {
            fprintf(f,"%d ",indices[i]);
            Camera camera = cameras(i);
            for ( int j = 0; j < 6; j++ )
            {
                fprintf(f,"%.15lf ",camera(j));
            }
            fprintf(f,"\n");
        }

        fclose(f);
    }
    
    void SfM::WritePointsOBJ( const std::string &path )
    {
        FILE *f = fopen( path.c_str(), "w" );

        std::vector<int> nobs(numPoints);
        std::vector<double> distances(numPoints);
        for ( int j = 0; j < numPoints; j++ )
        {
            nobs[j] = 0;
        }
        
        for ( int i = 0; i < numCameras; i++ )
        {
            Pose pose = GetPose(i);
            Eigen::Vector3d center = pose.getCenter();
            
            for ( int j = 0; j < numPoints; j++ )
            {
                if ( !points.exists(j) ) continue;
                if ( !observations.exists(i,j) ) continue;
                
                nobs[j]++;
                distances[j] = (GetPoint(j)-center).norm();
            }
        }
        
        for ( int i = 0; i < numPoints; i++ )
        {
            if ( !points.exists(i) ) continue;
            
            if ( distances[i] > 2000. ) continue;
            Point X = GetPoint(i);
            if ( X.norm() == 0 ) continue;
            fprintf(f,"v %0.15lf %0.15lf %0.15lf\n", X(0), X(1), X(2) );
        }
        
        fclose( f );
    }

    void SfM::WriteCameraCentersOBJ( const std::string &path )
    {
        FILE *f = fopen( path.c_str(), "w" );
        
        for ( int i = 0; i < numCameras; i++ )
        {
            Pose pose = GetPose(i);
            Eigen::Vector3d center = pose.getCenter();
            fprintf(f,"v %0.15lf %0.15lf %0.15lf\n", center(0), center(1), center(2) );
        }
        
        fclose( f );
    }
    
    void SfM::Normalize()
    {
        // calculate centroid 
        // shift so that centroid is at origin
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for ( int i = 0; i < GetNumCameras(); i++ )
        {
            Pose pose = GetPose(i);
            centroid += pose.getCenter();
        }
        centroid /= GetNumCameras();

        std::cout << "centroid over " << GetNumCameras() << " cameras: " << centroid.transpose() << "\n";
        Pose T( -centroid, Eigen::Vector3d::Zero() );
        Apply(T);

        // divide by average distance from origin
        double avg_scale = 0;
        for ( int i = 0; i < GetNumCameras(); i++ )
        {
            Pose pose = GetPose(i);
            avg_scale += pose.getCenter().norm();
        }
        avg_scale /= GetNumCameras();

        std::cout << "average scale over " << GetNumCameras() << " cameras: " << avg_scale << "\n";
        Apply(1./avg_scale);
        
        // check if reconstruction has inverted
        if ( GetPose(0).t(2) > 0 )
        {
            std::cout << "inverted! flipping to correct\n";
            std::cout << "first camera translation was: " << GetPose(0).t.transpose() << "\n";
            Apply(-1);
            std::cout << "first camera translation is now: " << GetPose(0).t.transpose() << "\n";
        }
    }
    
    void SfM::WriteCOLMAP( const std::string &path, int width, int height )
    {
        std::string sparse_dir = path + "/sparse";
        mkdir(sparse_dir.c_str(),0777);

        // write cameras.txt
        std::string cameras_path = sparse_dir + "/cameras.txt";
        FILE *camerasf = fopen(cameras_path.c_str(),"w");
        fprintf(camerasf,"# Camera list with one line of data per camera:\n");
        fprintf(camerasf,"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
        fprintf(camerasf,"# Number of cameras: 1\n");
        fprintf(camerasf,"1 SIMPLE_PINHOLE %d %d %lf %lf %lf\n",
            width, height, intrinsics.focal, intrinsics.centerx, intrinsics.centery );
        fclose(camerasf);

        // write images.txt
        std::string images_path = sparse_dir + "/images.txt";
        FILE *imagesf = fopen(images_path.c_str(),"w");
        fprintf(imagesf,"# Image list with two lines of data per image:\n");
        fprintf(imagesf,"#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
        fprintf(imagesf,"#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
        fprintf(imagesf,"# Number of images: %d, mean observations per image:\n", GetNumCameras() );
        std::vector< std::vector< std::pair<int,int> > > point_obs(GetNumPoints());
        for ( int i = 0; i < GetNumCameras(); i++ )
        {
            Pose pose = GetPose(i);
            fprintf(imagesf,"%d ",i+1);
            Eigen::Quaterniond quat(Eigen::Matrix3d::Identity());
            if ( pose.r.norm() != 0 )
            {
                Eigen::AngleAxisd angleaxis(pose.r.norm(),pose.r/pose.r.norm());
                quat = Eigen::Quaterniond(angleaxis);
            }
            fprintf(imagesf,"%lf %lf %lf %lf ",quat.w(),quat.x(),quat.y(),quat.z());
            fprintf(imagesf,"%lf %lf %lf ",pose.t(0),pose.t(1),pose.t(2));
            fprintf(imagesf,"1 ");
            fprintf(imagesf,"%s\n",paths(i).c_str());
            int k = 0;
            for ( int j = 0; j < GetNumPoints(); j++ )
            {
                Eigen::Vector3d point = GetPoint(j);
                if ( point.norm() == 0 ) continue;
                Observation obs;
                if ( !GetObservation(i,j,obs) ) continue;
                fprintf(imagesf,"%lf %lf %d ",obs(0)+intrinsics.centerx,obs(1)+intrinsics.centery,j+1);
                point_obs[j].push_back(std::make_pair(i+1,k));
                k++;
            }
            fprintf(imagesf,"\n");
        }
        fclose(imagesf);
        
        // write points3D.txt
        std::string points_path = sparse_dir + "/points3D.txt";
        FILE *pointsf = fopen(points_path.c_str(),"w");
        fprintf(pointsf,"# 3D point list with one line of data per point:\n");
        fprintf(pointsf,"#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
        fprintf(pointsf,"# Number of points: %d, mean track length: \n",GetNumPoints());
        for ( int j = 0; j < GetNumPoints(); j++ )
        {
            Eigen::Vector3d point = GetPoint(j);
            if ( point.norm() == 0 ) continue;
            fprintf(pointsf,"%d ",j+1);
            fprintf(pointsf,"%lf %lf %lf ",point(0),point(1),point(2));
            fprintf(pointsf,"0 0 0 "); // RGB
            fprintf(pointsf,"0 "); // error
            for ( int k = 0; k < point_obs[j].size(); k++ )
            {
                fprintf(pointsf,"%d %d ",point_obs[j][k].first,point_obs[j][k].second);
            }
            fprintf(pointsf,"\n");
        }
        fclose(pointsf);
    }
}
