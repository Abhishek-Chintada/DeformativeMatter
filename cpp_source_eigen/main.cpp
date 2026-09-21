#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix3d K;
    K << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    
    Eigen::Vector3d F{10, 20, 30};
    Eigen::Vector3d u = K.colPivHouseholderQr().solve(F);
    
    std::cout << "Displacements:\n" << u << std::endl;
    return 0;
}