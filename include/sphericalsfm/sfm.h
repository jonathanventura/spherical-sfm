
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sphericalsfm/sparse.hpp>
#include <sphericalsfm/sfm_types.h>

#include <opencv2/core.hpp>

#include <ceres/problem.h>
#include <ceres/solver.h>

namespace sphericalsfm {
    struct ViewTriplet
    {
        int i, j, k;
        ViewTriplet() : i(-1),j(-1),k(-1) { }
        ViewTriplet(int _i, int _j, int _k) : i(_i), j(_j), k(_k) { }
    };
    typedef Eigen::Matrix<double,5,1> Essential; // first 3 are Rp, second 2 are R
    typedef Eigen::Vector3d Rotation;

    class SfM
    {
    protected:
        friend class ParallelTriangulator;
        friend class ParallelRobustTriangulator;
        Intrinsics intrinsics;
        
        SparseVector<Camera> cameras;          // m cameras
        SparseVector<Point> points;            // n points
        
        SparseMatrix<Observation> observations;  // m x n x 2
        SparseMatrix<Pose> measurements;  // m x m
        SparseMatrix<bool> edges;  // m x m
        //SparseMatrix<Essential> edges;  // m x m
        //SparseMatrix<Rotation> sphericalEdges;  // m x m
        std::vector<ViewTriplet> viewTriplets;
        
        SparseVector<std::string> paths;       // indexed by camera index
        SparseVector<cv::Mat> descriptors;     // indexed by point index
        
        SparseVector<bool> rotationFixed;
        SparseVector<bool> translationFixed;
        SparseVector<bool> pointFixed;
        
        int numCameras;
        int numPoints;
        
        int nextCamera;
        
        int nextPoint;
        
        double * GetCameraPtr( int camera );
        double * GetPointPtr( int point );
        double * GetEdgePtr( int i, int j );
        //double * GetSphericalEdgePtr( int i, int j );
                
        ceres::LossFunction *loss_function;

        void PreOptimize();
        void ConfigureSolverOptions( ceres::Solver::Options &options, ceres::LinearSolverType linear_solver_type );
        void AddResidual( ceres::Problem &problem, int camera, int point );
        void AddViewTripletResidual( ceres::Problem &problem, int i, int j, int k, int point );
        void AddViewGraphResidual( ceres::Problem &problem, int i, int j, int point );
        //void AddSphericalViewGraphResidual( ceres::Problem &problem, int i, int j, int point );
        void AddPoseGraphResidual( ceres::Problem &problem, int i, int j );
        void PostOptimize();
    public:
        SfM( const Intrinsics &_intrinsics );
        
        Intrinsics GetIntrinsics() const { return intrinsics; }
        
        int AddCamera( const Pose &initial_pose, const std::string &path = "" );
        int AddPoint( const Point &initial_position, const cv::Mat &descriptor = cv::Mat() );
        void AddObservation( int camera, int point, const Observation &observation );
        void AddMeasurement( int i, int j, const Pose &measurement );
        void AddEdge( int i, int j );//, const Pose &relpose );
        //void AddSphericalEdge( int i, int j, const Rotation &rot );
        void AddViewTriplet( int i, int j, int k );

        void RemoveCamera( int camera );
        void RemovePoint( int point );
        
        void MergePoint( int point1, int point2 ); // point2 will be removed
        
        int GetNumCameras() { return numCameras; }
        int GetNumPoints() { return numPoints; }
        bool GetObservation( int camera, int point, Observation &observation );
        bool GetMeasurement( int i, int j, Pose &measurement );
        cv::Mat GetDescriptor( int point ) { return descriptors(point); }
        
        void Retriangulate();
        void RetriangulateRobust();
        
        bool Optimize();
        bool OptimizeViewTriplets();
        bool OptimizeViewGraph();
        //void ExtractPosesFromGraph();
        //bool OptimizeSphericalViewGraph();
        //void ExtractPosesFromSphericalViewGraph();
        
        void Apply( const Pose &pose );
        void Apply( double scale );
        void Unapply( const Pose &pose );
        
        Pose GetPose( int camera );
        void SetPose( int camera, const Pose &pose );
        Point GetPoint( int point );
        void SetPoint( int point, const Point &position );
        
        void SetRotationFixed( int camera, bool fixed ) { rotationFixed[camera] = fixed; }
        void SetTranslationFixed( int camera, bool fixed ) { translationFixed[camera] = fixed; }
        void SetPointFixed( int point, bool fixed ) { pointFixed[point] = fixed; }
        
        void WritePoses( const std::string &path, const std::vector<int> &indices );
        void WritePointsOBJ( const std::string &path );
        void WriteCameraCentersOBJ( const std::string &path );
        
        void Normalize();
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

