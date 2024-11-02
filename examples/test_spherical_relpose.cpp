#include <Eigen/Dense>
#include <sphericalsfm/so3.h>
#include <sphericalsfm/spherical_solvers.h>
#include <PoseLib/solvers/relpose_5pt.h>
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
    Eigen::Matrix3d &R, Eigen::Vector3d &t )
{
    // make random spherical relative pose problem
    // with specified relative rotation

    // make rotation and outward-facing translation
    R = sphericalsfm::so3exp(Eigen::Vector3d(0,angle,0));
    t = R.col(2)-Eigen::Vector3d(0,0,1);
    // R = sphericalsfm::so3exp(Eigen::Vector3d::Random());
    // t = Eigen::Vector3d::Random();

    x1.resize(6);
    x2.resize(6);

    for ( int i = 0; i < 6; i++ )
    {
        // make random 2D point in first image
        x1[i] = Eigen::Vector2d::Random();

        // make random depth from 4 to 8
        double depth = Eigen::Matrix<double,1,1>::Random()(0,0)*2+6;

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

bool compute_fivept( const std::vector<Eigen::Vector2d> &x1,
                     const std::vector<Eigen::Vector2d> &x2,
                     Eigen::Matrix3d &R, Eigen::Vector3d &t )
{
    std::vector<Eigen::Vector3d> x1h(5);
    std::vector<Eigen::Vector3d> x2h(5);
    for ( int i = 0; i < 5; i++ ) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
    }

    // compute essential matrix solutions from five pt solver
    std::vector<Eigen::Matrix3d> essential_matrices;
    int nsols = poselib::relpose_5pt(x1h, x2h, &essential_matrices);
    if ( !nsols ) return false;

    // compute sampson error for each solution
    int best_sol = 0;
    double best_score = INFINITY;
    for ( int i = 0; i < nsols; i++ )
    {
        std::vector<double> residuals;
        ComputeSquaredSampsonError(x1, x2, essential_matrices[i], &residuals);
        double score = std::accumulate(residuals.begin(),residuals.end(),0.);
        if ( score < best_score )
        {
            best_score = score;
            best_sol = i;
        }
    }

    Eigen::Matrix3d E = essential_matrices[best_sol];

    Rigid3d cam2_from_cam1;
    std::vector<Eigen::Vector3d> points3D;
    PoseFromEssentialMatrix(E, x1, x2, &cam2_from_cam1, &points3D);

    R = cam2_from_cam1.rotation.toRotationMatrix();
    t = cam2_from_cam1.translation;

    return true;
}

void compute_threept( Eigen::Vector2d x[5], Eigen::Vector2d y[5],
                     Eigen::Matrix3d R, Eigen::Vector3d t )
{
    
}

double compute_error( const Eigen::Vector3d &t1, const Eigen::Vector3d &t2 )
{
    double err = t1.dot(t2)/t1.norm()/t2.norm();
    return acos(std::min(std::max(err,-1.0),1.0)); 
}

int main( int argc, char **argv )
{
    for ( int iter = 0; iter < 1000; iter++ )
    {
        // std::cout << "**** iteration " << iter << " **** \n";

        double angle = 10.*M_PI/180., noise = 1/400;

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

        // compute five-point relative pose
        Eigen::Matrix3d R_fivept;
        Eigen::Vector3d t_fivept;
        if ( !compute_fivept(x1, x2, R_fivept, t_fivept ) ) continue;
        Eigen::Matrix3d E_fivept = sphericalsfm::skew3(t_fivept)*R_fivept;
        Eigen::Vector3d c_fivept = -R_fivept.transpose() * t_fivept;

        double err_fivept = compute_error(t,t_fivept);
        // std::cout << "E:\n" << E/E.norm() << "\n";
        // std::cout << "E_fivept:\n" << E_fivept/E_fivept.norm() << "\n";
        // std::cout << "t: " << t.transpose()/t.norm() << "\n";
        // std::cout << "t_fivept: " << t_fivept.transpose()/t_fivept.norm() << "\n";
        // std::cout << (c - c_fivept).norm() << "\n";
        std::cout << "fivepoint error: " << err_fivept << "\n";
        // std::cout << "fivepoint acos error: " << acos(err_fivept) << "\n";

        // compute spherical three-point relative pose
        // spherical_solver_action_matrix(const RayPairList &correspondences, const std::vector<int>& sample, std::vector<Eigen::Matrix3d>* Es);
    }
}