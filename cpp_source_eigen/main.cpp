#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <vector>
#include <cstdio>

using Eigen::Matrix3d;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::VectorXd;

typedef Eigen::Triplet<double> T;

struct Mesh {
    std::vector<Vector2d> nodes;
    std::vector<std::array<int, 3>> tris;
};



int main(void) {

    return 0;
}
