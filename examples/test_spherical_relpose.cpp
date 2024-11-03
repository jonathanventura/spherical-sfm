#include <Eigen/Dense>
#include <sphericalsfm/so3.h>
#include <sphericalsfm/spherical_solvers.h>
#include <sphericalsfm/spherical_utils.h>
#include <PoseLib/solvers/relpose_5pt.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <numeric>
#include <iostream>

struct Rigid3d {
 public:
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Rigid3d() = default;
  Rigid3d(const Eigen::Quaterniond& rotation,
          const Eigen::Vector3d& translation)
      : rotation(rotation), translation(translation) {}

  inline Eigen::Matrix<double,3,4> ToMatrix() const {
    Eigen::Matrix<double,3,4> matrix;
    matrix.leftCols<3>() = rotation.toRotationMatrix();
    matrix.col(3) = translation;
    return matrix;
  }

};


bool make_random_problem( double angle, double noise,
    std::vector<Eigen::Vector2d> &x1, std::vector<Eigen::Vector2d> &x2,
    Eigen::Matrix3d &R, Eigen::Vector3d &t,
    int num_observations = 15 )
{
    // make random spherical relative pose problem
    // with specified relative rotation

    // make rotation and outward-facing translation
    R = sphericalsfm::so3exp(Eigen::Vector3d(0,angle,0));
    t = R.col(2)-Eigen::Vector3d(0,0,1);
    // R = sphericalsfm::so3exp(Eigen::Vector3d::Random());
    // t = Eigen::Vector3d::Random();

    x1.resize(num_observations);
    x2.resize(num_observations);

    for ( int i = 0; i < num_observations; i++ )
    {
        // make random 2D point in first image
        x1[i] = Eigen::Vector2d::Random();

        // make random depth
        // double depth = Eigen::Matrix<double,1,1>::Random()(0,0)*2+4;
        // double depth = Eigen::Matrix<double,1,1>::Random()(0,0)*0.5+1;
        // double depth = Eigen::Matrix<double,1,1>::Random()(0,0)*1+11;
        double depth = Eigen::Matrix<double,1,1>::Random()(0,0)*9.5+10.5;
        if ( depth < 0 ) return false;

        // get 3D point
        Eigen::Vector3d X = depth * x1[i].homogeneous();

        // transform points to second camera's coordinate system
        Eigen::Vector3d PX = R*X+t;

        // check depth in second camera
        if ( PX(2) < 0 ) return false;

        // project point to second camera
        x2[i] = PX.head(2)/PX(2);

        // add noise to observations
        if ( noise > 0 )
        {
            x1[i] += Eigen::Vector2d::Random()*noise;
            x2[i] += Eigen::Vector2d::Random()*noise;
        }
    }

    return true;
} 

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals) {
  const size_t num_points1 = points1.size();
  residuals->resize(num_points1);
  for (size_t i = 0; i < num_points1; ++i) {
    const Eigen::Vector3d epipolar_line1 = E * points1[i].homogeneous();
    const Eigen::Vector3d point2_homogeneous = points2[i].homogeneous();
    const double num = point2_homogeneous.dot(epipolar_line1);
    const Eigen::Vector4d denom(point2_homogeneous.dot(E.col(0)),
                                point2_homogeneous.dot(E.col(1)),
                                epipolar_line1.x(),
                                epipolar_line1.y());
    (*residuals)[i] = num * num / denom.squaredNorm();
  }
}

bool TriangulatePoint(const Eigen::Matrix<double,3,4>& cam1_from_world,
                      const Eigen::Matrix<double,3,4>& cam2_from_world,
                      const Eigen::Vector2d& point1,
                      const Eigen::Vector2d& point2,
                      Eigen::Vector3d* xyz) {
  Eigen::Matrix4d A;
  A.row(0) = point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
  A.row(1) = point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
  A.row(2) = point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
  A.row(3) = point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  if (svd.matrixV()(3, 3) == 0) {
    return false;
  }

  *xyz = svd.matrixV().col(3).hnormalized();
  return true;
}

double CalculateDepth(const Eigen::Matrix<double,3,4>& cam_from_world,
                      const Eigen::Vector3d& point3D) {
  const double proj_z = cam_from_world.row(2).dot(point3D.homogeneous());
  return proj_z * cam_from_world.col(2).norm();
}

bool CheckCheirality(const Rigid3d& cam2_from_cam1,
                     const std::vector<Eigen::Vector2d>& points1,
                     const std::vector<Eigen::Vector2d>& points2,
                     std::vector<Eigen::Vector3d>* points3D) {
  const Eigen::Matrix<double,3,4> cam1_from_world = Eigen::Matrix<double,3,4>::Identity();
  const Eigen::Matrix<double,3,4> cam2_from_world = cam2_from_cam1.ToMatrix();
  constexpr double kMinDepth = std::numeric_limits<double>::epsilon();
  const double max_depth = 1000.0 * cam2_from_cam1.translation.norm();
  points3D->clear();
  for (size_t i = 0; i < points1.size(); ++i) {
    Eigen::Vector3d point3D;
    if (!TriangulatePoint(cam1_from_world,
                          cam2_from_world,
                          points1[i],
                          points2[i],
                          &point3D)) {
    //   std::cout << "\t\tpoint " << i << ": could not triangulate\n";
      continue;
    }
    const double depth1 = CalculateDepth(cam1_from_world, point3D);
    if (depth1 < kMinDepth || depth1 > max_depth) {
    //   std::cout << "\t\tpoint " << i << ": depth1 out of range\n";
      continue;
    }
    const double depth2 = CalculateDepth(cam2_from_world, point3D);
    if (depth2 < kMinDepth || depth2 > max_depth) {
    //   std::cout << "\t\tpoint " << i << ": depth2 out of range\n";
      continue;
    }
    //   std::cout << "\t\tpoint " << i << ": good\n";
    points3D->push_back(point3D);
  }
  return !points3D->empty();
}

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2,
                              Eigen::Vector3d* t) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV().transpose();

  if (U.determinant() < 0) {
    U *= -1;
  }
  if (V.determinant() < 0) {
    V *= -1;
  }

  Eigen::Matrix3d W;
  W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

  *R1 = U * W * V;
  *R2 = U * W.transpose() * V;
  *t = U.col(2).normalized();
}

void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Rigid3d* cam2_from_cam1,
                             std::vector<Eigen::Vector3d>* points3D) {

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d t;
  DecomposeEssentialMatrix(E, &R1, &R2, &t);

  const Eigen::Quaterniond quat1(R1);
  const Eigen::Quaterniond quat2(R2);

  // Generate all possible pose combinations.
  const std::array<Rigid3d, 4> cams2_from_cams1{{Rigid3d(quat1, t),
                                                 Rigid3d(quat2, t),
                                                 Rigid3d(quat1, -t),
                                                 Rigid3d(quat2, -t)}};
  
  points3D->clear();
  std::vector<Eigen::Vector3d> tentative_points3D;
  for (size_t i = 0; i < cams2_from_cams1.size(); ++i) {
    // std::cout << "R" << i << ":\n" << cams2_from_cams1[i].rotation.toRotationMatrix() << "\n";
    CheckCheirality(cams2_from_cams1[i], points1, points2, &tentative_points3D);
    // std::cout << "\tcount:" << tentative_points3D.size() << "\n";
    if (tentative_points3D.size() >= points3D->size()) {
      *cam2_from_cam1 = cams2_from_cams1[i];
      std::swap(*points3D, tentative_points3D);
    }
  }
}

Eigen::Matrix3d choose_best_Esoln( const std::vector<Eigen::Vector2d> &x1,
                                   const std::vector<Eigen::Vector2d> &x2,
                                   const std::vector<Eigen::Matrix3d> Es )
{
    // compute sampson error for each solution
    int best_sol = 0;
    double best_score = INFINITY;
    for ( int i = 0; i < Es.size(); i++ )
    {
        std::vector<double> residuals;
        ComputeSquaredSampsonError(x1, x2, Es[i], &residuals);
        double score = std::accumulate(residuals.begin(),residuals.end(),0.);
        if ( score < best_score )
        {
            best_score = score;
            best_sol = i;
        }
    }

    return Es[best_sol];
}

bool compute_nister( const std::vector<Eigen::Vector2d> &x1,
                     const std::vector<Eigen::Vector2d> &x2,
                     std::vector<Eigen::Matrix3d> &Es,
                     Eigen::Matrix3d &R, Eigen::Vector3d &t )
{
    std::vector<Eigen::Vector3d> x1h(5);
    std::vector<Eigen::Vector3d> x2h(5);
    for ( int i = 0; i < 5; i++ ) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
        x1h[i] /= x1h[i].norm();
        x2h[i] /= x2h[i].norm();
    }

    // compute essential matrix solutions from five pt solver
    int nsols = poselib::relpose_5pt(x1h, x2h, &Es);
    if ( !nsols ) return false;

    Eigen::Matrix3d E = choose_best_Esoln(x1,x2,Es);

    Rigid3d cam2_from_cam1;
    std::vector<Eigen::Vector3d> points3D;
    PoseFromEssentialMatrix(E, x1, x2, &cam2_from_cam1, &points3D);

    R = cam2_from_cam1.rotation.toRotationMatrix();
    t = cam2_from_cam1.translation;

    return true;
}

bool compute_stewenius( const std::vector<Eigen::Vector2d> &x1,
                        const std::vector<Eigen::Vector2d> &x2,
                        std::vector<Eigen::Matrix3d> &Es,
                        Eigen::Matrix3d &R, Eigen::Vector3d &t )
{
    opengv::bearingVectors_t x1h(5);
    opengv::bearingVectors_t x2h(5);
    for ( int i = 0; i < 5; i++ ) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
        x1h[i] /= x1h[i].norm();
        x2h[i] /= x2h[i].norm();
    }
    opengv::relative_pose::CentralRelativeAdapter adapter( x1h, x2h );



    opengv::complexEssentials_t fivept_stewenius_essentials = opengv::relative_pose::fivept_stewenius( adapter );


    if ( fivept_stewenius_essentials.empty() ) return false;
    Es.resize(fivept_stewenius_essentials.size());
    for ( int i = 0; i < fivept_stewenius_essentials.size(); i++ ) {
        Es[i] = fivept_stewenius_essentials[i].real().transpose();
    }
    Eigen::Matrix3d E = choose_best_Esoln(x1,x2,Es);

    Rigid3d cam2_from_cam1;
    std::vector<Eigen::Vector3d> points3D;
    PoseFromEssentialMatrix(E, x1, x2, &cam2_from_cam1, &points3D);

    R = cam2_from_cam1.rotation.toRotationMatrix();
    t = cam2_from_cam1.translation;

    return true;
}

bool compute_threept( const std::vector<Eigen::Vector2d> &x1,
                      const std::vector<Eigen::Vector2d> &x2,
                      std::vector<Eigen::Matrix3d> &Es,
                      Eigen::Matrix3d &R, Eigen::Vector3d &t )
{
    sphericalsfm::RayPairList correspondences(3);
    std::vector<int> sample(3);
    for ( int i = 0; i < 3; i++ )
    {
        correspondences[i].first = x1[i].homogeneous();
        correspondences[i].second = x2[i].homogeneous();
        correspondences[i].first /= correspondences[i].first.norm();
        correspondences[i].second /= correspondences[i].second.norm();
        sample[i] = i;
    }
    int nsols = sphericalsfm::spherical_solver_action_matrix(correspondences,sample,&Es);
    if ( !nsols ) return false;

    Eigen::Matrix3d E = choose_best_Esoln(x1,x2,Es);
    Eigen::Vector3d r;
    sphericalsfm::decompose_spherical_essential_matrix( E, false, r, t );
    R = sphericalsfm::so3exp(r);
    return true;
}

Eigen::Matrix3d normalize_E( const Eigen::Matrix3d &E )
{
    Eigen::Map<const Eigen::Matrix<double,9,1> > v(E.data());
    return E / v.norm();
}

double compute_essential_error( const Eigen::Matrix3d &E1, const Eigen::Matrix3d &E2 )
{
    return (normalize_E(E1)-normalize_E(E2)).norm();
}

double compute_rot_error( const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2 )
{
    return sphericalsfm::so3ln(R1*R2.transpose()).norm();
}

double compute_trans_error( const Eigen::Vector3d &t1, const Eigen::Vector3d &t2 )
{
    double err = t1.dot(t2)/t1.norm()/t2.norm();
    return acos(std::min(std::max(err,-1.0),1.0)); 
}

int main( int argc, char **argv )
{
    double focal = 600;
    FILE *f = fopen("results.csv","w");
    fprintf(f,"angle,noise,E_err,R_err,t_err,method\n");
    // for ( int iangle = 1; iangle > 0; iangle-- )
    int iangle = 1;
    double angle = 1;
    for ( int inoise = 0; inoise <= 10; inoise++ )
    {
        // double angle = iangle*M_PI/180./10;
        for ( int iter = 0; iter < 1000; iter++ )
        {
            // std::cout << "**** iteration " << iter << " **** \n";

            // double noise = 0;
            double noise = inoise/focal;

            std::vector<Eigen::Vector2d> x1;
            std::vector<Eigen::Vector2d> x2;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            // make data
            while ( !make_random_problem(angle, noise, x1, x2, R, t) ) ;
            Eigen::Vector3d c = -R.transpose() * t;
            Eigen::Matrix3d E = sphericalsfm::skew3(t)*R;

            // std::cout << "R:\n" << R << "\n";
            // std::cout << "t:\n" << t << "\n";

            // compute nister five-point relative pose
            Eigen::Matrix3d R_nister;
            Eigen::Vector3d t_nister;
            std::vector<Eigen::Matrix3d> Es_nister;
            if ( !compute_nister(x1, x2, Es_nister, R_nister, t_nister ) ) continue;
            Eigen::Matrix3d E_nister = sphericalsfm::skew3(t_nister)*R_nister;
            Eigen::Vector3d c_nister = -R_nister.transpose() * t_nister;

            double E_err_nister = compute_essential_error(E,E_nister);
            double R_err_nister = compute_rot_error(R,R_nister);
            double t_err_nister = compute_trans_error(t,t_nister);
            // std::cout << "E:\n" << normalize_E(E) << "\n";
            // for ( int i = 0; i < Es_nister.size(); i++ )
            // {
            //     std::cout << "E_nister " << i << ":\n" << normalize_E(Es_nister[i]) << "\n";
            // }
            // std::cout << "t: " << t.transpose()/t.norm() << "\n";
            // std::cout << "t_nister: " << t_nister.transpose()/t_nister.norm() << "\n";
            // std::cout << (c - c_nister).norm() << "\n";
            // std::cout << "fivepoint E error: " << E_err_nister << "\n";
            // std::cout << "fivepoint t error: " << t_err_nister << "\n";
            // std::cout << "fivepoint acos error: " << acos(err_nister) << "\n";

            // compute stewenius five-point relative pose
            Eigen::Matrix3d R_stewenius;
            Eigen::Vector3d t_stewenius;
            std::vector<Eigen::Matrix3d> Es_stewenius;
            if ( !compute_stewenius(x1, x2, Es_stewenius, R_stewenius, t_stewenius ) ) continue;
            Eigen::Matrix3d E_stewenius = sphericalsfm::skew3(t_stewenius)*R_stewenius;
            // Eigen::Vector3d c_stewenius = -R_stewenius.transpose() * t_stewenius;

            double E_err_stewenius = compute_essential_error(E,E_stewenius);
            double R_err_stewenius = compute_rot_error(R,R_stewenius);
            double t_err_stewenius = compute_trans_error(t,t_stewenius);

            // compute spherical three-point relative pose
            Eigen::Matrix3d R_threept;
            Eigen::Vector3d t_threept;
            std::vector<Eigen::Matrix3d> Es_threept;
            if ( !compute_threept(x1, x2, Es_threept, R_threept, t_threept ) ) continue;
            Eigen::Matrix3d E_threept = sphericalsfm::skew3(t_threept)*R_threept;
            double E_err_threept = compute_essential_error(E,E_threept);
            double R_err_threept = compute_rot_error(R,R_threept);
            double t_err_threept = compute_trans_error(t,t_threept);
            // std::cout << "threepoint E error: " << E_err_threept << "\n";
            // std::cout << "threepoint t error: " << t_err_threept << "\n";

            fprintf(f,"%d,%d,%f,%f,%f,nister\n",iangle,inoise,E_err_nister,R_err_nister,t_err_nister);
            fprintf(f,"%d,%d,%f,%f,%f,stewenius\n",iangle,inoise,E_err_stewenius,R_err_stewenius,t_err_stewenius);
            fprintf(f,"%d,%d,%f,%f,%f,threept\n",iangle,inoise,E_err_threept,R_err_threept,t_err_threept);
        }
    }
    fclose(f);
}