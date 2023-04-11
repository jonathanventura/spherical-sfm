#include <Eigen/Core>
#include <vector>

// NE: 4x3x3 nullspace
// w: 3x1 nullspace coefficients
void solver_relpose_5p(Eigen::MatrixXd const& NE, std::vector<Eigen::MatrixXcd>* w);
