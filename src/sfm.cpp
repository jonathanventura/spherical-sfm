
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>

#include <Eigen/Jacobi>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <iostream>

#include <opencv2/core/utility.hpp>

#include <sphericalsfm/sfm.h>
#include <sphericalsfm/so3.h>

namespace sphericalsfm {

    class ParallelTriangulator : public cv::ParallelLoopBody
    {
    public:
        ParallelTriangulator( SfM &_sfm ) :sfm(_sfm) { }
        virtual void operator()(const cv::Range &range) const CV_OVERRIDE
        {
            for ( int j = range.start; j < range.end; j++ )
            {
                if ( !sfm.points.exists(j) ) continue;
                
                int firstcam = -1;
                int lastcam = -1;
                
                int nobs = 0;
                for ( int i = 0; i < sfm.numCameras; i++ )
                {
                    if ( !sfm.cameras.exists(i) ) continue;
                    if ( !( sfm.observations.exists(i,j) ) ) continue;
                    if ( firstcam == -1 ) firstcam = i;
                    lastcam = i;
                    nobs++;
                }

                sfm.SetPoint( j, Eigen::Vector3d::Zero() );
                if ( nobs < 3 ) continue;

                Eigen::MatrixXd A( nobs*2, 4 );
                 
                int n = 0;
                for ( int i = 0; i < sfm.numCameras; i++ )
                {
                    if ( !sfm.cameras.exists(i) ) continue;
                    if ( !( sfm.observations.exists(i,j) ) ) continue;

                    Observation vec = sfm.observations(i,j);

                    Eigen::Vector2d point(vec(0)/sfm.intrinsics.focal,vec(1)/sfm.intrinsics.focal);
                    Eigen::Matrix4d P = sfm.GetPose(i).P;
                    
                    A.row(2*n+0) = P.row(2) * point[0] - P.row(0);
                    A.row(2*n+1) = P.row(2) * point[1] - P.row(1);
                    n++;
                }

                Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A,Eigen::ComputeFullV);
                Eigen::Vector4d Xh = svdA.matrixV().col(3);
                Eigen::Vector3d X = Xh.head(3)/Xh(3);

                sfm.SetPoint( j, X );
            }
        }
        ParallelTriangulator& operator=(const ParallelTriangulator &) {
            return *this;
        }
    private:
        SfM &sfm;
    };
    
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

    void SfM::AddMeasurement( int i, int j, const Pose &measurement )
    {
        measurements(i,j) = measurement;
    }

    bool SfM::GetMeasurement( int i, int j, Pose &measurement )
    {
        if ( !measurements.exists(i,j) ) return false;

        measurement = measurements(i,j);
        
        return true;
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
        //for ( int j = 0; j < numPoints; j++ )
        for ( int j = range.start; j < range.end; j++ )
        {
            if ( !points.exists(j) ) continue;
            
            int firstcam = -1;
            int lastcam = -1;
            
            int nobs = 0;
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !cameras.exists(i) ) continue;
                if ( !( observations.exists(i,j) ) ) continue;
                if ( firstcam == -1 ) firstcam = i;
                lastcam = i;
                nobs++;
            }

            SetPoint( j, Eigen::Vector3d::Zero() );
            if ( nobs < 3 ) continue;

            Eigen::MatrixXd A( nobs*2, 4 );
             
            int n = 0;
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !cameras.exists(i) ) continue;
                if ( !( observations.exists(i,j) ) ) continue;

                Observation vec = observations(i,j);

                Eigen::Vector2d point(vec(0)/intrinsics.focal,vec(1)/intrinsics.focal);
                Eigen::Matrix4d P = GetPose(i).P;
                
                A.row(2*n+0) = P.row(2) * point[0] - P.row(0);
                A.row(2*n+1) = P.row(2) * point[1] - P.row(1);
                n++;
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A,Eigen::ComputeFullV);
            Eigen::Vector4d Xh = svdA.matrixV().col(3);
            Eigen::Vector3d X = Xh.head(3)/Xh(3);

            SetPoint( j, X );
        }
        });
    }

    void SfM::PreOptimize()
    {
        loss_function = new ceres::CauchyLoss( 2.0 );
    }

    void SfM::ConfigureSolverOptions( ceres::Solver::Options &options )
    {
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = 1000;
        options.max_num_consecutive_invalid_steps = 100;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 16;
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
        ConfigureSolverOptions( options );
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        if ( summary.termination_type == ceres::FAILURE )
        {
            std::cout << "error: ceres failed.\n";
            exit(1);
        }
        
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
            Point X = GetPoint(j);
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
            Point X = GetPoint(j);
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
}
