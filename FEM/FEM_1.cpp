// Minimal 2D FEM (constant-strain triangles, plane stress, linear elastic)
// Build: g++ -O2 -std=c++17 -I /usr/include/eigen3 fem2d.cpp -o fem2d
// Test: cantilever, left edge fixed, downward point-load distributed on right edge.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::VectorXd;
typedef Eigen::Triplet<double> T;

struct Mesh {
    std::vector<Vector2d> nodes;
    std::vector<std::array<int, 3>> tris;
};

// Structured nx-by-ny grid on [0,L]x[0,H], each cell split into 2 triangles.
Mesh makeGrid(double L, double H, int nx, int ny) {
    Mesh m;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
            m.nodes.emplace_back(L * i / nx, H * j / ny);
    auto id = [&](int i, int j) { return j * (nx + 1) + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            // counter-clockwise ordering matters (positive area)
            m.tris.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
            m.tris.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
        }
    return m;
}

// Plane-stress constitutive matrix D
Matrix3d planeStressD(double E, double nu) {
    Matrix3d D;
    double c = E / (1.0 - nu * nu);
    D << 1, nu, 0,
         nu, 1, 0,
         0, 0, (1 - nu) / 2;
    return c * D;
}

// B matrix (3x6) and area for one triangle
Matrix<double, 3, 6> bMatrix(const Mesh& m, const std::array<int, 3>& t, double& area) {
    const Vector2d &p1 = m.nodes[t[0]], &p2 = m.nodes[t[1]], &p3 = m.nodes[t[2]];
    double b[3] = {p2.y() - p3.y(), p3.y() - p1.y(), p1.y() - p2.y()};
    double c[3] = {p3.x() - p2.x(), p1.x() - p3.x(), p2.x() - p1.x()};
    area = 0.5 * (p1.x() * b[0] + p2.x() * b[1] + p3.x() * b[2]);
    Matrix<double, 3, 6> B;
    B.setZero();
    for (int k = 0; k < 3; ++k) {
        B(0, 2 * k) = b[k];
        B(1, 2 * k + 1) = c[k];
        B(2, 2 * k) = c[k];
        B(2, 2 * k + 1) = b[k];
    }
    return B / (2.0 * area);
}

int main() {
    const double L = 10.0, H = 1.0, thick = 1.0;
    const double E = 200e3, nu = 0.3;  // MPa-ish units, keep consistent!
    const int nx = 40, ny = 4;
    const double P = -10.0;            // total downward load on right edge

    Mesh mesh = makeGrid(L, H, nx, ny);
    const int N = (int)mesh.nodes.size(), ndof = 2 * N;
    Matrix3d D = planeStressD(E, nu);

    // --- Assemble global stiffness ---
    std::vector<T> trip;
    for (auto& t : mesh.tris) {
        double A;
        auto B = bMatrix(mesh, t, A);
        Matrix<double, 6, 6> Ke = thick * A * B.transpose() * D * B;
        int dof[6] = {2 * t[0], 2 * t[0] + 1, 2 * t[1], 2 * t[1] + 1, 2 * t[2], 2 * t[2] + 1};
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) trip.emplace_back(dof[a], dof[b], Ke(a, b));
    }
    Eigen::SparseMatrix<double> K(ndof, ndof);
    K.setFromTriplets(trip.begin(), trip.end());

    // --- Loads: spread P over right-edge nodes ---
    VectorXd f = VectorXd::Zero(ndof);
    std::vector<int> right;
    for (int i = 0; i < N; ++i)
        if (std::abs(mesh.nodes[i].x() - L) < 1e-9) right.push_back(i);
    for (int n : right) f(2 * n + 1) = P / right.size();  // crude but fine for a first test

    // --- Boundary conditions: fix left edge (eliminate rows/cols) ---
    std::vector<bool> fixed(ndof, false);
    for (int i = 0; i < N; ++i)
        if (std::abs(mesh.nodes[i].x()) < 1e-9) fixed[2 * i] = fixed[2 * i + 1] = true;

    std::vector<int> freeMap(ndof, -1);
    int nf = 0;
    for (int d = 0; d < ndof; ++d)
        if (!fixed[d]) freeMap[d] = nf++;

    std::vector<T> tf;
    for (int k = 0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it)
            if (freeMap[it.row()] >= 0 && freeMap[it.col()] >= 0)
                tf.emplace_back(freeMap[it.row()], freeMap[it.col()], it.value());
    Eigen::SparseMatrix<double> Kff(nf, nf);
    Kff.setFromTriplets(tf.begin(), tf.end());
    VectorXd ff(nf);
    for (int d = 0; d < ndof; ++d)
        if (freeMap[d] >= 0) ff(freeMap[d]) = f(d);

    // --- Solve ---
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(Kff);
    if (solver.info() != Eigen::Success) { std::puts("Factorization failed"); return 1; }
    VectorXd uf = solver.solve(ff);
    VectorXd u = VectorXd::Zero(ndof);
    for (int d = 0; d < ndof; ++d)
        if (freeMap[d] >= 0) u(d) = uf(freeMap[d]);

    // --- Sanity check against Euler-Bernoulli: delta = P L^3 / (3 E I) ---
    double I = thick * H * H * H / 12.0;
    double analytic = P * L * L * L / (3 * E * I);
    double tip = 0;
    for (int n : right) tip += u(2 * n + 1);
    tip /= right.size();
    std::printf("tip deflection FEM: %.6f   beam theory: %.6f\n", tip, analytic);

    // --- Output for plotting / OpenGL: node positions + displacements, element von Mises ---
    FILE* fn = std::fopen("nodes.csv", "w");
    for (int i = 0; i < N; ++i)
        std::fprintf(fn, "%f,%f,%f,%f\n", mesh.nodes[i].x(), mesh.nodes[i].y(), u(2 * i), u(2 * i + 1));
    std::fclose(fn);

    FILE* fe = std::fopen("elements.csv", "w");
    for (auto& t : mesh.tris) {
        double A;
        auto B = bMatrix(mesh, t, A);
        Matrix<double, 6, 1> ue;
        for (int k = 0; k < 3; ++k) { ue(2 * k) = u(2 * t[k]); ue(2 * k + 1) = u(2 * t[k] + 1); }
        Eigen::Vector3d s = D * B * ue;  // sxx, syy, txy
        double vm = std::sqrt(s(0) * s(0) - s(0) * s(1) + s(1) * s(1) + 3 * s(2) * s(2));
        std::fprintf(fe, "%d,%d,%d,%f\n", t[0], t[1], t[2], vm);
    }
    std::fclose(fe);
    return 0;
}
