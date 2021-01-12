
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

#include <RansacLib/ransac.h>
#include <sphericalsfm/triangulation_estimator.h>

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

    struct ViewTripletError
    {
        ViewTripletError( double _focal, double _xi, double _yi, double _xj, double _yj, double _xk, double _yk )
        : focal(_focal), xi(_xi), yi(_yi), xj(_xj), yj(_yj), xk(_xk), yk(_yk)
        {
            
        }
        
        template <typename T>
        bool operator()(const T* const camerai_t,
                        const T* const camerai_r,
                        const T* const cameraj_t,
                        const T* const cameraj_r,
                        const T* const camerak_t,
                        const T* const camerak_r,
                        T* residuals) const
        {
            // construct essential matrices
            T Ri_data[9];
            T Rj_data[9];
            T Rk_data[9];
            ceres::AngleAxisToRotationMatrix(camerai_r,Ri_data);
            ceres::AngleAxisToRotationMatrix(cameraj_r,Rj_data);
            ceres::AngleAxisToRotationMatrix(camerak_r,Rk_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Ri(Ri_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rj(Rj_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rk(Rk_data);
            Eigen::Map< const Eigen::Matrix<T,3,1> > ti(camerai_t);
            Eigen::Map< const Eigen::Matrix<T,3,1> > tj(cameraj_t);
            Eigen::Map< const Eigen::Matrix<T,3,1> > tk(camerak_t);
            
            Eigen::Matrix<T,3,3> Rik = Rk * Ri.transpose();
            Eigen::Matrix<T,3,1> tik = Rk * (-Ri.transpose()*ti) + tk;
            Eigen::Matrix<T,3,3> Rjk = Rk * Rj.transpose();
            Eigen::Matrix<T,3,1> tjk = Rk * (-Rj.transpose()*tj) + tk;
            Eigen::Matrix<T,3,3> Eik = skew3(tik)*Rik;
            Eigen::Matrix<T,3,3> Ejk = skew3(tjk)*Rjk;
            Eigen::Matrix<T,3,1> xih;
            xih << T(xi),T(yi),T(focal);
            Eigen::Matrix<T,3,1> xjh;
            xjh << T(xj),T(yj),T(focal);
            Eigen::Matrix<T,3,1> xkhat = (Eik * xih).cross(Ejk * xjh);

            // projection
            T xp = xkhat(0) / xkhat(2);
            T yp = xkhat(1) / xkhat(2);
            
            // intrinsics
            T fxp = T(focal) * xp;
            T fyp = T(focal) * yp;
            
            // residuals
            residuals[0] = fxp - T(xk);
            residuals[1] = fyp - T(yk);
            
            return true;
        }
        
        double focal, xi, yi, xj, yj, xk, yk;
    };

/*
    struct ViewGraphError
    {
        ViewGraphError( double _focal, double _xi, double _yi, double _xj, double _yj, double _xk, double _yk )
        : focal(_focal), xi(_xi), yi(_yi), xj(_xj), yj(_yj), xk(_xk), yk(_yk)
        {
            
        }
        
        template <typename T>
        bool operator()(const T* const essentialik,
                        const T* const essentialjk,
                        T* residuals) const
        {
            Eigen::Matrix<T,3,3> C;
            C << T(0), T(1), T(0),
            T(-1), T(0), T(0),
            T(0), T(0), T(0);

            const T* const Rpik_vec = essentialik;
            T Rpik_data[9];
            ceres::AngleAxisToRotationMatrix(Rpik_vec,Rpik_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rpik(Rpik_data);

            T Rik_vec[3] = { essentialik[3], essentialik[4], T(0) };
            T Rik_data[9];
            ceres::AngleAxisToRotationMatrix(Rik_vec,Rik_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rik(Rik_data);

            const T* const Rpjk_vec = essentialjk;
            T Rpjk_data[9];
            ceres::AngleAxisToRotationMatrix(Rpjk_vec,Rpjk_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rpjk(Rpjk_data);

            T Rjk_vec[3] = { essentialjk[3], essentialjk[4], T(0) };
            T Rjk_data[9];
            ceres::AngleAxisToRotationMatrix(Rjk_vec,Rjk_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rjk(Rjk_data);
            
            Eigen::Matrix<T,3,3> Eik = Rpik * C *Rik.transpose();
            Eigen::Matrix<T,3,3> Ejk = Rpjk * C *Rjk.transpose();

            Eigen::Matrix<T,3,1> xih;
            xih << T(xi),T(yi),T(focal);
            Eigen::Matrix<T,3,1> xjh;
            xjh << T(xj),T(yj),T(focal);
            Eigen::Matrix<T,3,1> xkhat = (Eik * xih).cross(Ejk * xjh);

            // projection
            T xp = xkhat(0) / xkhat(2);
            T yp = xkhat(1) / xkhat(2);
            
            // intrinsics
            T fxp = T(focal) * xp;
            T fyp = T(focal) * yp;
            
            // residuals
            residuals[0] = fxp - T(xk);
            residuals[1] = fyp - T(yk);
            
            return true;
        }
        
        double focal, xi, yi, xj, yj, xk, yk;
    };
*/

    struct ViewGraphError
    {
        ViewGraphError( double _focal, double _xi, double _yi, double _xj, double _yj )
        : focal(_focal), xi(_xi), yi(_yi), xj(_xj), yj(_yj)
        {
            
        }
        
        template <typename T>
        bool operator()(const T* const ti_data,
                        const T* const ri_data,
                        const T* const tj_data,
                        const T* const rj_data,
                        T* residuals) const
        {
            Eigen::Map< const Eigen::Matrix<T,3,1> > ti(ti_data);

            T Ri_data[9];
            ceres::AngleAxisToRotationMatrix(ri_data,Ri_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Ri(Ri_data);

            Eigen::Map< const Eigen::Matrix<T,3,1> > tj(tj_data);

            T Rj_data[9];
            ceres::AngleAxisToRotationMatrix(rj_data,Rj_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rj(Rj_data);
            
            Eigen::Matrix<T,3,3> R = Rj*Ri.transpose();
            Eigen::Matrix<T,3,1> t = Rj*(-Ri.transpose()*ti) + tj;
            
            Eigen::Matrix<T,3,3> E = skew3(t) * R;

            Eigen::Matrix<T,3,1> u, v;
            u << T(xi),T(yi),T(focal);
            v << T(xj),T(yj),T(focal);
            Eigen::Matrix<T,3,1> line = E * (u/u(2));
            T d = v.dot( line );
        
            residuals[0] = (T(focal)*d) / sqrt(line[0]*line[0] + line[1]*line[1]);
            
            return true;
        }
        
        double focal, xi, yi, xj, yj;
    };

    struct SphericalViewGraphError
    {
        SphericalViewGraphError( double _focal, double _xi, double _yi, double _xj, double _yj )
        : focal(_focal), xi(_xi), yi(_yi), xj(_xj), yj(_yj)
        {
            
        }
        
        template <typename T>
        bool operator()(const T* const ri,
                        const T* const rj,
                        T* residuals) const
        {
            T Ri_data[9];
            ceres::AngleAxisToRotationMatrix(ri,Ri_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Ri(Ri_data);

            T Rj_data[9];
            ceres::AngleAxisToRotationMatrix(rj,Rj_data);
            Eigen::Map< const Eigen::Matrix<T,3,3> > Rj(Rj_data);
            
            Eigen::Matrix<T,3,3> R = Rj*Ri.transpose();
            
            Eigen::Matrix<T,3,1> z;
            z << T(0),T(0),T(1);
            Eigen::Matrix<T,3,1> t = R.col(2) - z;
            Eigen::Matrix<T,3,3> E = skew3(t) * R;

            Eigen::Matrix<T,3,1> u, v;
            u << T(xi),T(yi),T(focal);
            v << T(xj),T(yj),T(focal);
            Eigen::Matrix<T,3,1> line = E * (u/u(2));
            T d = v.dot( line );
        
            //residuals[0] = (d*d) / (line[0]*line[0] + line[1]*line[1]);
            //residuals[0] *= T(focal)*T(focal);
            residuals[0] = (T(focal)*d) / sqrt(line[0]*line[0] + line[1]*line[1]);
            
            return true;
        }
        
        double focal, xi, yi, xj, yj;
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

    double * SfM::GetEdgePtr( int i, int j )
    {
        return (double*)&edges(i,j);
    }

/*
    double * SfM::GetSphericalEdgePtr( int i, int j )
    {
        return (double*)&sphericalEdges(i,j);
    }
*/

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

    void SfM::AddViewTriplet( int i, int j, int k )
    {
        viewTriplets.push_back(ViewTriplet(i,j,k));
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

/*
    void SfM::AddEdge( int i, int j, const Pose &relpose )
    {
        Eigen::Matrix3d R = so3exp(relpose.r);
        Eigen::Vector3d t = relpose.t / relpose.t.norm();
        
        std::cout << "R:\n " << R << "\n";
        std::cout << "t:\n " << t.transpose() << "\n";
        std::cout << "skew3(t):\n " << skew3(t) << "\n";
        Eigen::Matrix3d E = skew3(t)*R;

        Eigen::Vector3d c = -R.transpose() * t;
        Eigen::Vector3d axis = c.cross(Eigen::Vector3d(0,0,1));
        axis /= axis.norm();
        double angle = acos(c.dot(Eigen::Vector3d(0,0,1)));
        Eigen::Matrix3d T = so3exp(axis*angle);

        Eigen::Matrix3d newR = R*T.transpose();
        
        // Rp is newR
        // R is T'
        Eigen::Vector3d rp = so3ln(newR);
        Eigen::Vector3d r = so3ln(T.transpose());
        Essential e;
        e << rp[0],rp[1],rp[2],r[0],r[1];
        
        std::cout << "rp: " << rp.transpose() << " r: " << r.transpose() << "\n";

        Eigen::Matrix3d C;
        C << 0,1,0,
        -1,0,0,
        0,0,0;
        Eigen::Matrix3d newE = newR*C*T;
        std::cout << "old E:\n" << E << "\nnew E:\n" << newE << "\n";

        std::cout << "e: " << e.transpose() << "\n";
        
        edges(i,j) = e;
    }
*/
    void SfM::AddEdge( int i, int j )
    {
        edges(i,j) = true;
    }

/*
    void SfM::AddSphericalEdge( int i, int j, const Rotation &rot )
    {
        sphericalEdges(i,j) = rot;
    }
*/

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
            
            SetPoint( j, X );
        }
        });
    }
/*
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
*/

    void SfM::RetriangulateRobust()
    {
        cv::parallel_for_(cv::Range(0,numPoints), [&](const cv::Range &range){
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

            const double min_depth = 0.1;
            const double max_depth = 1000.;
            const double min_disp = 1./max_depth;
            const double max_disp = 1./min_depth;
            const int nsteps = 128;
            Pose refpose = GetPose(firstcam);
            Observation refobs = observations(firstcam,j);
            Eigen::Vector3d bestX = Eigen::Vector3d::Zero();
            double bestscore = INFINITY;
            for ( int n = 0; n < nsteps; n++ )
            {
                double disp = min_disp + (max_disp-min_disp)*(double)n/(nsteps-1);
                Eigen::Vector3d X(refobs(0),refobs(1),intrinsics.focal);
                X /= disp;
                X = refpose.applyInverse(X);
            
                double score = 0;
                for ( int i = 0; i < numCameras; i++ )
                {
                    if ( !cameras.exists(i) ) continue;
                    if ( !( observations.exists(i,j) ) ) continue;

                    Pose pose = GetPose(i);
                    Eigen::Vector3d PX = pose.apply(X);
                    Eigen::Vector2d proj = PX.head(2)/PX(2);
                    Observation obs = observations(i,j);
                
                    Eigen::Vector2d diff = intrinsics.focal*proj-obs;
                    double err = diff.norm();
                    score += std::max(err,2.);
                }
                if ( score < bestscore )
                {
                    bestscore = score; 
                    bestX = X;
                }
            }

            SetPoint( j, bestX );
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

    void SfM::AddViewGraphResidual( ceres::Problem &problem, int i, int j, int point )
    {
        Observation veci = observations(i,point);
        Observation vecj = observations(j,point);
        
        ViewGraphError *reproj_error = new ViewGraphError(intrinsics.focal,veci(0),veci(1),vecj(0),vecj(1));
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ViewGraphError, 1, 3, 3, 3, 3>(reproj_error);
        problem.AddResidualBlock(cost_function, loss_function,
            GetCameraPtr(i), GetCameraPtr(i)+3,
            GetCameraPtr(j), GetCameraPtr(j)+3
        );

        if ( translationFixed( i ) ) problem.SetParameterBlockConstant( GetCameraPtr(i) );
        if ( rotationFixed( i ) ) problem.SetParameterBlockConstant( GetCameraPtr(i)+3 );
        if ( translationFixed( j ) ) problem.SetParameterBlockConstant( GetCameraPtr(j) );
        if ( rotationFixed( j ) ) problem.SetParameterBlockConstant( GetCameraPtr(j)+3 );
    }

/*
    void SfM::AddSphericalViewGraphResidual( ceres::Problem &problem, int i, int j, int point )
    {
        Observation veci = observations(i,point);
        Observation vecj = observations(j,point);
        
        SphericalViewGraphError *reproj_error = new SphericalViewGraphError(intrinsics.focal,veci(0),veci(1),vecj(0),vecj(1));
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SphericalViewGraphError, 1, 3, 3>(reproj_error);
        problem.AddResidualBlock(cost_function, loss_function,
            GetCameraPtr(i)+3,  GetCameraPtr(j)+3
        );
        if ( rotationFixed( i ) ) problem.SetParameterBlockConstant( GetCameraPtr(i)+3 );
        if ( rotationFixed( j ) ) problem.SetParameterBlockConstant( GetCameraPtr(j)+3 );
    }
*/

    void SfM::AddViewTripletResidual( ceres::Problem &problem, int i, int j, int k, int point )
    {
        Observation veci = observations(i,point);
        Observation vecj = observations(j,point);
        Observation veck = observations(k,point);
        
        ViewTripletError *reproj_error = new ViewTripletError(intrinsics.focal,veci(0),veci(1),vecj(0),vecj(1),veck(0),veck(1));
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ViewTripletError, 2, 3, 3, 3, 3, 3, 3>(reproj_error);
        problem.AddResidualBlock(cost_function, loss_function,
            GetCameraPtr(i), GetCameraPtr(i)+3,
            GetCameraPtr(j), GetCameraPtr(j)+3,
            GetCameraPtr(k), GetCameraPtr(k)+3
        );

        if ( translationFixed( i ) ) problem.SetParameterBlockConstant( GetCameraPtr(i) );
        if ( rotationFixed( i ) ) problem.SetParameterBlockConstant( GetCameraPtr(i)+3 );
        if ( translationFixed( j ) ) problem.SetParameterBlockConstant( GetCameraPtr(j) );
        if ( rotationFixed( j ) ) problem.SetParameterBlockConstant( GetCameraPtr(j)+3 );
        if ( translationFixed( k ) ) problem.SetParameterBlockConstant( GetCameraPtr(k) );
        if ( rotationFixed( k ) ) problem.SetParameterBlockConstant( GetCameraPtr(k)+3 );
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

    bool SfM::OptimizeViewTriplets()
    {
        if ( numCameras == 0 || numPoints == 0 ) return false;
        
        ceres::Problem problem;
        
        PreOptimize();
        
        bool added_one_camera = false;
        
        std::cout << "\tBuilding BA problem...\n";

        /*
        int ntriplets = 0;
        for ( int n = 0; n < viewTriplets.size(); n++ )
        {
            const ViewTriplet &trip = viewTriplets[n];
            int i = trip.i;
            int j = trip.j;
            int k = trip.k;
            for ( int pt = 0; pt < numPoints; pt++ )
            {
                if ( !points.exists(pt) ) continue;
            
                if ( !( observations.exists(i,pt) ) ) continue;
                if ( !( observations.exists(j,pt) ) ) continue;
                if ( !( observations.exists(k,pt) ) ) continue;
                        
                AddViewTripletResidual(problem,i,j,k,pt);
                AddViewTripletResidual(problem,i,k,j,pt);
                AddViewTripletResidual(problem,j,k,i,pt);
                added_one_camera = true;
                ntriplets++;
            }
        }
        std::cout << "\tadded " << ntriplets << " triplets\n";
        */

        int ntriplets = 0;
        for ( int pt = 0; pt < numPoints; pt++ )
        {
            std::cout << "point " << pt << "/" << numPoints << "\n";
            if ( !points.exists(pt) ) continue;
            for ( int i = 0; i < numCameras; i++ )
            {
                if ( !( observations.exists(i,pt) ) ) continue;
                for ( int j = i+1; j < numCameras; j++ )
                {
                    if ( !( observations.exists(j,pt) ) ) continue;
                    for ( int k = j+1; k < numCameras; k++ )
                    {
                        if ( !( observations.exists(k,pt) ) ) continue;
                    
                        AddViewTripletResidual(problem,i,j,k,pt);
                        AddViewTripletResidual(problem,i,k,j,pt);
                        AddViewTripletResidual(problem,j,k,i,pt);
                        added_one_camera = true;
                        ntriplets+=3;
                    }
                }
            }
        }
        std::cout << "\tadded " << ntriplets << " triplets\n";
        
        if ( !added_one_camera ) {
            std::cout << "didn't add any cameras\n";
            return false;
        }
        
        std::cout << "Running optimizer...\n";
        std::cout << "\t" << problem.NumResiduals() << " residuals\n";

        ceres::Solver::Options options;
        ConfigureSolverOptions( options, ceres::DENSE_NORMAL_CHOLESKY );
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

/*
    bool SfM::OptimizeViewGraph()
    {
        ceres::Problem problem;
        
        PreOptimize();
        
        bool added_one_edge = false;
        
        std::cout << "\tBuilding BA problem...\n";

        int ntriplets = 0;
        for ( int n = 0; n < viewTriplets.size(); n++ )
        {
            const ViewTriplet &trip = viewTriplets[n];
            int i = trip.i;
            int j = trip.j;
            int k = trip.k;
            for ( int pt = 0; pt < points.size(); pt++ )
            {
                if ( !points.exists(pt) ) continue;
            
                if ( !( observations.exists(i,pt) ) ) continue;
                if ( !( observations.exists(j,pt) ) ) continue;
                if ( !( observations.exists(k,pt) ) ) continue;

                if ( ( edges.exists(i,k) && edges.exists(j,k) ) )
                {
                    AddViewGraphResidual(problem,i,j,k,pt);
                    added_one_edge = true;
                }

                if ( ( edges.exists(i,j) && edges.exists(k,j) ) )
                {
                    AddViewGraphResidual(problem,i,k,j,pt);
                    added_one_edge = true;
                }

                if ( ( edges.exists(j,i) && edges.exists(k,i) ) )
                {
                    AddViewGraphResidual(problem,j,k,i,pt);
                    added_one_edge = true;
                }

                ntriplets++;
            }
        }
        std::cout << "\tadded " << ntriplets << " triplets\n";
        
        if ( !added_one_edge ) {
            std::cout << "didn't add any edges\n";
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
*/

    bool SfM::OptimizeViewGraph()
    {
        ceres::Problem problem;
        
        PreOptimize();
        
        bool added_one_edge = false;
        
        std::cout << "\tBuilding BA problem...\n";

        for ( int i = 0; i < cameras.size(); i++ )
        {
            std::cout << "\tcamera " << i << " / " << cameras.size()-1 << "\n";
            for ( int j = i+1; j < cameras.size(); j++ )
            {
                //if ( !edges.exists(i,j) ) continue;
                for ( int pt = 0; pt < points.size(); pt++ )
                {
                    if ( !points.exists(pt) ) continue;
                
                    if ( !( observations.exists(i,pt) ) ) continue;
                    if ( !( observations.exists(j,pt) ) ) continue;

                    AddViewGraphResidual(problem,i,j,pt);
                    added_one_edge = true;
                }
            }
        }
        
        if ( !added_one_edge ) {
            std::cout << "didn't add any edges\n";
            return false;
        }
        
        std::cout << "Running optimizer...\n";
        std::cout << "\t" << problem.NumResiduals() << " residuals\n";

        ceres::Solver::Options options;
        ConfigureSolverOptions( options, ceres::DENSE_NORMAL_CHOLESKY );
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        if ( summary.termination_type == ceres::FAILURE )
        {
            std::cout << "error: ceres failed.\n";
            exit(1);
        }
        
        PostOptimize();

        /*
        for ( int i = 0; i < cameras.size(); i++ )
        {
            for ( int j = i+1; j < cameras.size(); j++ )
            {
                if ( i > 30 || j < cameras.size()-30 ) continue;
                std::cout << i << "\t" << j << "\n";

                for ( int pt = 0; pt < points.size(); pt++ )
                {
                    if ( !points.exists(pt) ) continue;
                
                    if ( !( observations.exists(i,pt) ) ) continue;
                    std::cout << observations.exists(i,pt) << "\t" <<
                    observations.exists(j,pt) << "\n";
                    if ( !( observations.exists(j,pt) ) ) continue;

                    Observation veci = observations(i,pt);
                    Observation vecj = observations(j,pt);
        
                    SphericalViewGraphError *reproj_error = new SphericalViewGraphError(intrinsics.focal,veci(0),veci(1),vecj(0),vecj(1));
                    double resid;
                    (*reproj_error)(GetCameraPtr(i)+3, GetCameraPtr(j)+3,&resid);
                    
                    std::cout << i << "\t" << j << "\t" << pt << "\t" << resid << "\n";                    
                }
            }
        }
        */
        
        return ( summary.termination_type == ceres::CONVERGENCE );
    }

/*
    void SfM::ExtractPosesFromGraph()
    {
        // extract rotations
        for ( int i = 1; i < cameras.size(); i++ )
        {
            Essential e = edges(i-1,i);
            
        }

        // build system of linear constraints
        SetPose( 0, Pose() );
    }

    void SfM::ExtractPosesFromSphericalViewGraph()
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t(0,0,-1);

        SetPose( 0, Pose(t,so3ln(R)) );

        // extract rotations
        for ( int i = 1; i < cameras.size(); i++ )
        {
            Rotation r = sphericalEdges(i-1,i);
            R = so3exp(r)*R;
            SetPose( i, Pose(t,so3ln(R)) );
        }
    }
*/

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
    }
}
