// SOP

// fem_sheet.cpp
// Interactive 2D finite element sheet: bilinear quadrilateral (Q4) elements, plane stress / plane strain,
// optional hole, selectable thickness, movable load, nodal stress smoothing and an OpenGL contour map.
//
// Libraries: Eigen (linear algebra), GLFW (window + input), OpenGL 1.x fixed-function (drawing).
// Units: mm, N, MPa (N/mm^2). Keep them consistent.
//
// Run modes:
//   ./fem_sheet                  interactive viewer
//   ./fem_sheet --selftest       solve a cantilever and compare with beam theory (no window)
//   ./fem_sheet --shot out.ppm   render one frame to a PPM image and exit

#include <GLFW/glfw3.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::VectorXd;
typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 2, 4> Mat24;
typedef Matrix<double, 3, 8> Mat38;
typedef Matrix<double, 4, 2> Mat42;
typedef Matrix<double, 8, 8> Mat88;

// ============================================================================================
// 1. Parameters (everything the user can change while the program runs)
// ============================================================================================
enum Field { VON_MISES, SIGMA_XX, SIGMA_YY, TAU_XY, PRINC_MAX, PRINC_MIN, NFIELDS };
static const char* FIELD_NAMES[NFIELDS] = {"von Mises", "sigma_xx", "sigma_yy", "tau_xy",
                                           "max principal", "min principal"};
static const char* BC_NAMES[3] = {"left edge fixed", "left+right edges fixed", "bottom edge fixed"};
static const int MESH_LEVELS[4][2] = {{20, 10}, {40, 20}, {80, 40}, {120, 60}};

struct Params {
    double W = 200.0, H = 100.0;     // sheet width and height [mm]
    int meshLevel = 1;               // index into MESH_LEVELS
    int nx = 40, ny = 20;            // cells along x and y
    double thickness = 5.0;          // [mm]
    double E = 200e3, nu = 0.3;      // steel [MPa]
    bool planeStress = true;
    bool hole = true;
    double holeR = 20.0;             // [mm], hole is centred in the sheet
    int bcMode = 0;
    double Fmag = 1000.0;            // [N]
    double angleDeg = -90.0;         // force direction, -90 = straight down
    double lx = 200.0, ly = 50.0;    // load position [mm]
    double loadRadius = 8.0;         // load is spread over nodes within this radius [mm]
    bool loadEdge = false;           // selftest only: spread load over the whole right edge
    int field = VON_MISES;
    double defMult = 1.0;            // multiplier on the automatic deformation exaggeration
    int clipIdx = 0;                 // colour bar upper limit: 100%, 50%, 25%, 10% of the range
    bool banded = false;             // smooth or banded (contour line style) colouring
    bool showMesh = true, showUndeformed = true;
};

// ============================================================================================
// 2. Mesh
// ============================================================================================
struct Mesh {
    std::vector<Vector2d> X;                  // node coordinates (only nodes used by some element)
    std::vector<std::array<int, 4>> quads;    // 4 node ids per element, counter-clockwise
};

void buildMesh(const Params& p, Mesh& m) {
    m.X.clear();
    m.quads.clear();
    const double dx = p.W / p.nx, dy = p.H / p.ny;
    auto g = [&](int i, int j) { return j * (p.nx + 1) + i; };
    std::vector<int> newId((p.nx + 1) * (p.ny + 1), -1);
    std::vector<std::array<int, 4>> cells;
    for (int j = 0; j < p.ny; ++j)
        for (int i = 0; i < p.nx; ++i) {
            double cx = (i + 0.5) * dx, cy = (j + 0.5) * dy;
            if (p.hole && std::hypot(cx - p.W / 2, cy - p.H / 2) < p.holeR) continue;  // skip hole cells
            cells.push_back({g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)});
        }
    for (auto& c : cells) {
        std::array<int, 4> q;
        for (int k = 0; k < 4; ++k) {
            int old = c[k];
            if (newId[old] < 0) {  // first time this grid point is used: create the node
                newId[old] = (int)m.X.size();
                m.X.emplace_back((old % (p.nx + 1)) * dx, (old / (p.nx + 1)) * dy);
            }
            q[k] = newId[old];
        }
        m.quads.push_back(q);
    }
}

// ============================================================================================
// 3. Q4 element: shape functions, B matrix, stiffness
// ============================================================================================
static const double GP = 0.5773502691896257;               // 1/sqrt(3), Gauss point
static const double XI[4] = {-1, 1, 1, -1};                // natural coordinates of the 4 corners
static const double ETA[4] = {-1, -1, 1, 1};

// B matrix (3x8) at natural point (xi, eta) of an element with corner coordinates xy (4x2).
Mat38 bMatrix(const Mat42& xy, double xi, double eta, double& detJ) {
    Mat24 dN;  // derivatives of the shape functions w.r.t. xi (row 0) and eta (row 1)
    for (int k = 0; k < 4; ++k) {
        dN(0, k) = 0.25 * XI[k] * (1 + ETA[k] * eta);
        dN(1, k) = 0.25 * ETA[k] * (1 + XI[k] * xi);
    }
    Eigen::Matrix2d J = dN * xy;           // Jacobian: natural -> physical
    detJ = J.determinant();
    Mat24 dNxy = J.inverse() * dN;         // derivatives w.r.t. x and y
    Mat38 B = Mat38::Zero();
    for (int k = 0; k < 4; ++k) {
        B(0, 2 * k) = dNxy(0, k);
        B(1, 2 * k + 1) = dNxy(1, k);
        B(2, 2 * k) = dNxy(1, k);
        B(2, 2 * k + 1) = dNxy(0, k);
    }
    return B;
}

Matrix3d constitutive(const Params& p) {
    Matrix3d D;
    if (p.planeStress) {
        double c = p.E / (1 - p.nu * p.nu);
        D << 1, p.nu, 0, p.nu, 1, 0, 0, 0, (1 - p.nu) / 2;
        return c * D;
    }
    double c = p.E / ((1 + p.nu) * (1 - 2 * p.nu));
    D << 1 - p.nu, p.nu, 0, p.nu, 1 - p.nu, 0, 0, 0, (1 - 2 * p.nu) / 2;
    return c * D;
}

// ============================================================================================
// 4. FEM system: assemble, constrain, factorize, solve, recover stress
// ============================================================================================
struct Fem {
    Mesh mesh;
    Matrix3d D;
    std::vector<char> fixedDof;
    std::vector<int> freeMap;
    int nf = 0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    bool factored = false;
    VectorXd u;                                 // displacements [u0,v0,u1,v1,...]
    std::vector<double> sxx, syy, txy;          // smoothed nodal stress components
    std::vector<double> field;                  // chosen scalar for colouring
    double fmin = 0, fmax = 1, umax = 0;
    int loadedNodes = 0;
};

bool assemble(const Params& p, Fem& F) {
    const Mesh& m = F.mesh;
    const int N = (int)m.X.size(), nd = 2 * N;
    F.D = constitutive(p);

    std::vector<Trip> trip;
    trip.reserve(m.quads.size() * 64);
    for (auto& q : m.quads) {
        Mat42 xy;
        for (int k = 0; k < 4; ++k) { xy(k, 0) = m.X[q[k]].x(); xy(k, 1) = m.X[q[k]].y(); }
        Mat88 Ke = Mat88::Zero();
        for (double a : {-GP, GP})
            for (double b : {-GP, GP}) {  // 2x2 Gauss integration, all weights = 1
                double detJ;
                Mat38 B = bMatrix(xy, a, b, detJ);
                Ke += p.thickness * detJ * B.transpose() * F.D * B;
            }
        int dof[8];
        for (int k = 0; k < 4; ++k) { dof[2 * k] = 2 * q[k]; dof[2 * k + 1] = 2 * q[k] + 1; }
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b) trip.emplace_back(dof[a], dof[b], Ke(a, b));
    }
    Eigen::SparseMatrix<double> K(nd, nd);
    K.setFromTriplets(trip.begin(), trip.end());

    // Boundary conditions
    const double eps = 1e-9;
    F.fixedDof.assign(nd, 0);
    for (int n = 0; n < N; ++n) {
        double x = m.X[n].x(), y = m.X[n].y();
        bool fix = false;
        if (p.bcMode == 0) fix = x < eps;
        if (p.bcMode == 1) fix = x < eps || x > p.W - eps;
        if (p.bcMode == 2) fix = y < eps;
        if (fix) F.fixedDof[2 * n] = F.fixedDof[2 * n + 1] = 1;
    }
    F.freeMap.assign(nd, -1);
    F.nf = 0;
    for (int d = 0; d < nd; ++d)
        if (!F.fixedDof[d]) F.freeMap[d] = F.nf++;

    std::vector<Trip> tf;
    for (int k = 0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it)
            if (F.freeMap[it.row()] >= 0 && F.freeMap[it.col()] >= 0)
                tf.emplace_back(F.freeMap[it.row()], F.freeMap[it.col()], it.value());
    Eigen::SparseMatrix<double> Kff(F.nf, F.nf);
    Kff.setFromTriplets(tf.begin(), tf.end());
    F.ldlt.compute(Kff);
    F.factored = (F.ldlt.info() == Eigen::Success);
    if (!F.factored) std::puts("warning: factorization failed (check boundary conditions)");
    return F.factored;
}

void solveLoad(const Params& p, Fem& F) {
    if (!F.factored) return;
    const Mesh& m = F.mesh;
    const int N = (int)m.X.size(), nd = 2 * N;
    VectorXd f = VectorXd::Zero(nd);

    std::vector<int> nodes;
    if (p.loadEdge) {
        for (int n = 0; n < N; ++n)
            if (m.X[n].x() > p.W - 1e-9) nodes.push_back(n);
    } else {
        int nearest = 0;
        double best = 1e300;
        for (int n = 0; n < N; ++n) {
            double d = (m.X[n] - Vector2d(p.lx, p.ly)).norm();
            if (d < best) { best = d; nearest = n; }
            if (d <= p.loadRadius) nodes.push_back(n);
        }
        if (nodes.empty()) nodes.push_back(nearest);
    }
    F.loadedNodes = (int)nodes.size();
    double a = p.angleDeg * M_PI / 180.0;
    double Fx = p.Fmag * std::cos(a), Fy = p.Fmag * std::sin(a);
    for (int n : nodes) { f(2 * n) += Fx / nodes.size(); f(2 * n + 1) += Fy / nodes.size(); }

    VectorXd ff(F.nf);
    for (int d = 0; d < nd; ++d)
        if (F.freeMap[d] >= 0) ff(F.freeMap[d]) = f(d);
    VectorXd uf = F.ldlt.solve(ff);
    F.u = VectorXd::Zero(nd);
    for (int d = 0; d < nd; ++d)
        if (F.freeMap[d] >= 0) F.u(d) = uf(F.freeMap[d]);

    F.umax = 0;
    for (int n = 0; n < N; ++n) F.umax = std::max(F.umax, std::hypot(F.u(2 * n), F.u(2 * n + 1)));
}

// Evaluate stress at each element corner, then average over all elements touching the node.
void recoverStress(const Params& p, Fem& F) {
    (void)p;
    const Mesh& m = F.mesh;
    const int N = (int)m.X.size();
    F.sxx.assign(N, 0);
    F.syy.assign(N, 0);
    F.txy.assign(N, 0);
    std::vector<int> cnt(N, 0);
    for (auto& q : m.quads) {
        Mat42 xy;
        Matrix<double, 8, 1> ue;
        for (int k = 0; k < 4; ++k) {
            xy(k, 0) = m.X[q[k]].x(); xy(k, 1) = m.X[q[k]].y();
            ue(2 * k) = F.u(2 * q[k]); ue(2 * k + 1) = F.u(2 * q[k] + 1);
        }
        for (int k = 0; k < 4; ++k) {
            double detJ;
            Mat38 B = bMatrix(xy, XI[k], ETA[k], detJ);
            Eigen::Vector3d s = F.D * B * ue;
            F.sxx[q[k]] += s(0); F.syy[q[k]] += s(1); F.txy[q[k]] += s(2);
            cnt[q[k]]++;
        }
    }
    for (int n = 0; n < N; ++n)
        if (cnt[n]) { F.sxx[n] /= cnt[n]; F.syy[n] /= cnt[n]; F.txy[n] /= cnt[n]; }
}

void updateField(const Params& p, Fem& F) {
    const int N = (int)F.mesh.X.size();
    F.field.assign(N, 0);
    for (int n = 0; n < N; ++n) {
        double sx = F.sxx[n], sy = F.syy[n], t = F.txy[n];
        double sz = p.planeStress ? 0.0 : p.nu * (sx + sy);
        double R = std::sqrt(0.25 * (sx - sy) * (sx - sy) + t * t);
        double v = 0;
        switch (p.field) {
            case VON_MISES: v = std::sqrt(0.5 * ((sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) + (sz - sx) * (sz - sx)) + 3 * t * t); break;
            case SIGMA_XX: v = sx; break;
            case SIGMA_YY: v = sy; break;
            case TAU_XY: v = t; break;
            case PRINC_MAX: v = 0.5 * (sx + sy) + R; break;
            case PRINC_MIN: v = 0.5 * (sx + sy) - R; break;
        }
        F.field[n] = v;
    }
    F.fmin = *std::min_element(F.field.begin(), F.field.end());
    F.fmax = *std::max_element(F.field.begin(), F.field.end());
    if (F.fmax - F.fmin < 1e-12) F.fmax = F.fmin + 1e-12;
}

// ============================================================================================
// 5. Self test: slender cantilever vs Euler-Bernoulli
// ============================================================================================
int selftest() {
    Params p;
    p.W = 100; p.H = 10; p.nx = 100; p.ny = 8;
    p.hole = false; p.thickness = 1; p.Fmag = 10; p.angleDeg = -90; p.loadEdge = true; p.bcMode = 0;
    Fem F;
    buildMesh(p, F.mesh);
    assemble(p, F);
    solveLoad(p, F);
    double tip = 0; int c = 0;
    for (size_t n = 0; n < F.mesh.X.size(); ++n)
        if (F.mesh.X[n].x() > p.W - 1e-9) { tip += F.u(2 * n + 1); ++c; }
    tip /= c;
    double I = p.thickness * p.H * p.H * p.H / 12.0;
    double analytic = -p.Fmag * p.W * p.W * p.W / (3 * p.E * I);
    std::printf("nodes=%zu elements=%zu dof=%d\n", F.mesh.X.size(), F.mesh.quads.size(), (int)F.u.size());
    std::printf("tip deflection FEM (Q4): %.6f  beam theory: %.6f  ratio: %.3f\n", tip, analytic, tip / analytic);
    return 0;
}

// ============================================================================================
// 6. Rendering helpers
// ============================================================================================
static void colormap(double t, float& r, float& g, float& b) {  // classic "jet"
    auto cl = [](double v) { return (float)std::max(0.0, std::min(1.0, v)); };
    r = cl(1.5 - std::fabs(4 * t - 3));
    g = cl(1.5 - std::fabs(4 * t - 2));
    b = cl(1.5 - std::fabs(4 * t - 1));
}

static GLuint gTex = 0;
static void makeTexture(bool banded) {
    const int NT = 256, BANDS = 12;
    unsigned char img[NT * 3];
    for (int i = 0; i < NT; ++i) {
        double t = (i + 0.5) / NT;
        if (banded) t = (std::floor(t * BANDS) + 0.5) / BANDS;
        float r, g, b;
        colormap(t, r, g, b);
        img[3 * i] = (unsigned char)(255 * r); img[3 * i + 1] = (unsigned char)(255 * g); img[3 * i + 2] = (unsigned char)(255 * b);
    }
    if (!gTex) glGenTextures(1, &gTex);
    glBindTexture(GL_TEXTURE_1D, gTex);
    GLint filt = banded ? GL_NEAREST : GL_LINEAR;
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, filt);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, NT, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
}

// Tiny 7-segment text renderer so the colour bar can show numbers without any font library.
static void drawText(const char* s, double x, double y, double h) {
    static const int MASK[10] = {0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};  // bits: a b c d e f g
    double w = 0.55 * h;
    glBegin(GL_LINES);
    for (const char* c = s; *c; ++c) {
        int mask = -1;
        if (*c >= '0' && *c <= '9') mask = MASK[*c - '0'];
        else if (*c == '-') mask = 0x40;
        if (mask >= 0) {
            double X[7][4] = {{0, h, w, h}, {w, h, w, h / 2}, {w, h / 2, w, 0}, {0, 0, w, 0},
                              {0, h / 2, 0, 0}, {0, h, 0, h / 2}, {0, h / 2, w, h / 2}};
            for (int sgm = 0; sgm < 7; ++sgm)
                if (mask & (1 << sgm)) {
                    glVertex2d(x + X[sgm][0], y + X[sgm][1]);
                    glVertex2d(x + X[sgm][2], y + X[sgm][3]);
                }
        }
        if (*c == '.') {
            glVertex2d(x + 0.1 * h, y);
            glVertex2d(x + 0.1 * h, y + 0.08 * h);
            x += 0.3 * h;
        } else {
            x += w + 0.3 * h;
        }
    }
    glEnd();
}

// ============================================================================================
// 7. Application state, view transform, drawing
// ============================================================================================
struct App {
    Params p;
    Fem F;
    bool dirtyMesh = true, dirtyAssemble = true, dirtySolve = true, dirtyField = true, dirtyTitle = true;
    bool dirtyTexture = true;
    bool dragging = false;
    int ww = 1100, wh = 720;
} app;

struct View { double s, ox, oy; };
static View viewFor(int ww, int wh) {
    const double barW = 150;
    View v;
    v.s = std::min((ww - barW - 60) / (app.p.W * 1.1), (wh - 100) / (app.p.H * 1.7));
    v.ox = (ww - barW) / 2 - v.s * app.p.W / 2;
    v.oy = wh / 2 - v.s * app.p.H / 2;
    return v;
}

static void drawScene(GLFWwindow* win) {
    int ww, wh, fw, fh;
    glfwGetWindowSize(win, &ww, &wh);
    glfwGetFramebufferSize(win, &fw, &fh);
    app.ww = ww; app.wh = wh;
    glViewport(0, 0, fw, fh);
    glClearColor(0.97f, 0.97f, 0.97f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, ww, 0, wh, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);

    const Params& p = app.p;
    const Fem& F = app.F;
    View v = viewFor(ww, wh);
    const int N = (int)F.mesh.X.size();
    if (N == 0 || (int)F.u.size() != 2 * N || (int)F.field.size() != N) return;

    double autoScale = F.umax > 0 ? 0.08 * p.H / F.umax : 0.0;
    double def = autoScale * p.defMult;
    std::vector<Vector2d> P(N), P0(N);
    for (int n = 0; n < N; ++n) {
        Vector2d x0 = F.mesh.X[n];
        Vector2d x1 = x0 + def * Vector2d(F.u(2 * n), F.u(2 * n + 1));
        P0[n] = Vector2d(v.ox + v.s * x0.x(), v.oy + v.s * x0.y());
        P[n] = Vector2d(v.ox + v.s * x1.x(), v.oy + v.s * x1.y());
    }

    // contour map: scalar -> 1D texture coordinate
    double top = F.fmin + (F.fmax - F.fmin) * (p.clipIdx == 0 ? 1.0 : p.clipIdx == 1 ? 0.5 : p.clipIdx == 2 ? 0.25 : 0.1);
    auto tc = [&](int n) {
        double t = (F.field[n] - F.fmin) / (top - F.fmin);
        t = std::max(0.0, std::min(1.0, t));
        return (float)(0.5 / 256 + t * 255.0 / 256);
    };
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, gTex);
    glColor3f(1, 1, 1);
    glBegin(GL_TRIANGLES);
    for (auto& q : F.mesh.quads) {
        int idx[6] = {0, 1, 2, 0, 2, 3};
        for (int k : idx) { glTexCoord1f(tc(q[k])); glVertex2d(P[q[k]].x(), P[q[k]].y()); }
    }
    glEnd();
    glDisable(GL_TEXTURE_1D);

    if (p.showUndeformed) {
        glColor4f(0, 0, 0, 0.35f); glLineWidth(1.0f);
        for (auto& q : F.mesh.quads) {
            // only draw outer boundary edges of the undeformed shape: edge shared by 2 quads would be drawn twice, fine
            glBegin(GL_LINE_LOOP);
            for (int k = 0; k < 4; ++k) glVertex2d(P0[q[k]].x(), P0[q[k]].y());
            glEnd();
        }
    }
    if (p.showMesh) {
        glColor4f(0, 0, 0, 0.12f); glLineWidth(1.0f);
        for (auto& q : F.mesh.quads) {
            glBegin(GL_LINE_LOOP);
            for (int k = 0; k < 4; ++k) glVertex2d(P[q[k]].x(), P[q[k]].y());
            glEnd();
        }
    }

    // fixed nodes
    glColor3f(0.1f, 0.1f, 0.1f); glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int n = 0; n < N; ++n)
        if (F.fixedDof[2 * n]) glVertex2d(P0[n].x(), P0[n].y());
    glEnd();

    // load: patch circle + arrow ending at the load point
    double cx = v.ox + v.s * p.lx, cy = v.oy + v.s * p.ly;
    double a = p.angleDeg * M_PI / 180.0, dxn = std::cos(a), dyn = std::sin(a);
    glColor3f(0.1f, 0.1f, 0.1f); glLineWidth(2.5f);
    if (!p.loadEdge) {
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 40; ++i)
            glVertex2d(cx + v.s * p.loadRadius * std::cos(i * M_PI / 20), cy + v.s * p.loadRadius * std::sin(i * M_PI / 20));
        glEnd();
        double len = 70;
        double sx = cx - dxn * len, sy = cy - dyn * len;
        glBegin(GL_LINES);
        glVertex2d(sx, sy); glVertex2d(cx, cy);
        glEnd();
        glBegin(GL_TRIANGLES);  // arrow head
        glVertex2d(cx, cy);
        glVertex2d(cx - dxn * 14 + dyn * 6, cy - dyn * 14 - dxn * 6);
        glVertex2d(cx - dxn * 14 - dyn * 6, cy - dyn * 14 + dxn * 6);
        glEnd();
    }

    // colour bar
    double bx = ww - 110, bw = 24, by0 = 70, by1 = wh - 70;
    glEnable(GL_TEXTURE_1D);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord1f(0.5f / 256); glVertex2d(bx, by0); glVertex2d(bx + bw, by0);
    glTexCoord1f(255.5f / 256); glVertex2d(bx + bw, by1); glVertex2d(bx, by1);
    glEnd();
    glDisable(GL_TEXTURE_1D);
    glColor3f(0, 0, 0); glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2d(bx, by0); glVertex2d(bx + bw, by0); glVertex2d(bx + bw, by1); glVertex2d(bx, by1);
    glEnd();
    for (int i = 0; i <= 5; ++i) {
        double t = i / 5.0, y = by0 + t * (by1 - by0);
        glBegin(GL_LINES); glVertex2d(bx + bw, y); glVertex2d(bx + bw + 5, y); glEnd();
        char buf[32];
        double val = F.fmin + t * (top - F.fmin);
        std::snprintf(buf, sizeof buf, std::fabs(val) < 100 ? "%.2f" : "%.0f", val);
        drawText(buf, bx + bw + 9, y - 5, 10);
    }
    glLineWidth(1.0f);
}

static void updateTitle(GLFWwindow* win) {
    const Params& p = app.p;
    const Fem& F = app.F;
    double top = F.fmin + (F.fmax - F.fmin) * (p.clipIdx == 0 ? 1.0 : p.clipIdx == 1 ? 0.5 : p.clipIdx == 2 ? 0.25 : 0.1);
    char buf[512];
    std::snprintf(buf, sizeof buf,
        "FEM sheet | %s [MPa] range %.2f..%.2f | t=%.2f mm F=%.0f N ang=%.0f | %s | %s | %s | %dx%d | umax=%.4f mm | def x%.0f",
        FIELD_NAMES[p.field], F.fmin, top, p.thickness, p.Fmag, p.angleDeg, p.planeStress ? "plane stress" : "plane strain",
        BC_NAMES[p.bcMode], p.hole ? "hole" : "no hole", p.nx, p.ny, F.umax,
        F.umax > 0 ? 0.08 * p.H / F.umax * p.defMult : 0.0);
    glfwSetWindowTitle(win, buf);
}

static void saveShot(GLFWwindow* win, const char* path) {
    int fw, fh;
    glfwGetFramebufferSize(win, &fw, &fh);
    std::vector<unsigned char> px(fw * fh * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::printf("cannot write %s\n", path); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", fw, fh);
    for (int y = fh - 1; y >= 0; --y) std::fwrite(&px[(size_t)y * fw * 3], 1, (size_t)fw * 3, f);
    std::fclose(f);
    std::printf("saved %s\n", path);
}

// ============================================================================================
// 8. Input
// ============================================================================================
static void moveLoadTo(double mx, double my) {
    View v = viewFor(app.ww, app.wh);
    double x = (mx - v.ox) / v.s, y = (app.wh - my - v.oy) / v.s;
    app.p.lx = std::max(0.0, std::min(app.p.W, x));
    app.p.ly = std::max(0.0, std::min(app.p.H, y));
    app.dirtySolve = true; app.dirtyField = true; app.dirtyTitle = true;
}

static void onMouse(GLFWwindow* w, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    app.dragging = (action == GLFW_PRESS);
    if (app.dragging) { double mx, my; glfwGetCursorPos(w, &mx, &my); moveLoadTo(mx, my); }
}
static void onCursor(GLFWwindow*, double mx, double my) { if (app.dragging) moveLoadTo(mx, my); }

static void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action == GLFW_RELEASE) return;
    Params& p = app.p;
    bool toSolve = false, toAssemble = false, toMesh = false, toField = false;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_UP: p.thickness = std::min(100.0, p.thickness * 1.25); toAssemble = true; break;
        case GLFW_KEY_DOWN: p.thickness = std::max(0.2, p.thickness / 1.25); toAssemble = true; break;
        case GLFW_KEY_RIGHT: p.Fmag = std::min(1e7, p.Fmag * 1.25); toSolve = true; break;
        case GLFW_KEY_LEFT: p.Fmag = std::max(1.0, p.Fmag / 1.25); toSolve = true; break;
        case GLFW_KEY_Q: p.angleDeg -= 15; toSolve = true; break;
        case GLFW_KEY_E: p.angleDeg += 15; toSolve = true; break;
        case GLFW_KEY_COMMA: p.loadRadius = std::max(1.0, p.loadRadius / 1.25); toSolve = true; break;
        case GLFW_KEY_PERIOD: p.loadRadius = std::min(60.0, p.loadRadius * 1.25); toSolve = true; break;
        case GLFW_KEY_1: p.field = VON_MISES; toField = true; break;
        case GLFW_KEY_2: p.field = SIGMA_XX; toField = true; break;
        case GLFW_KEY_3: p.field = SIGMA_YY; toField = true; break;
        case GLFW_KEY_4: p.field = TAU_XY; toField = true; break;
        case GLFW_KEY_5: p.field = PRINC_MAX; toField = true; break;
        case GLFW_KEY_6: p.field = PRINC_MIN; toField = true; break;
        case GLFW_KEY_LEFT_BRACKET: p.defMult = std::max(0.05, p.defMult / 1.5); app.dirtyTitle = true; break;
        case GLFW_KEY_RIGHT_BRACKET: p.defMult = std::min(50.0, p.defMult * 1.5); app.dirtyTitle = true; break;
        case GLFW_KEY_H: p.hole = !p.hole; toMesh = true; break;
        case GLFW_KEY_B: p.bcMode = (p.bcMode + 1) % 3; toAssemble = true; break;
        case GLFW_KEY_S: p.planeStress = !p.planeStress; toAssemble = true; break;
        case GLFW_KEY_MINUS:
            if (p.meshLevel > 0) { p.meshLevel--; toMesh = true; } break;
        case GLFW_KEY_EQUAL:
            if (p.meshLevel < 3) { p.meshLevel++; toMesh = true; } break;
        case GLFW_KEY_K: p.clipIdx = (p.clipIdx + 1) % 4; app.dirtyTitle = true; break;
        case GLFW_KEY_C: p.banded = !p.banded; app.dirtyTexture = true; break;
        case GLFW_KEY_M: p.showMesh = !p.showMesh; break;
        case GLFW_KEY_U: p.showUndeformed = !p.showUndeformed; break;
        case GLFW_KEY_P: app.dirtyTitle = true; saveShot(w, "shot.ppm"); break;
        default: break;
    }
    p.nx = MESH_LEVELS[p.meshLevel][0];
    p.ny = MESH_LEVELS[p.meshLevel][1];
    if (toMesh) app.dirtyMesh = app.dirtyAssemble = app.dirtySolve = app.dirtyField = true;
    if (toAssemble) app.dirtyAssemble = app.dirtySolve = app.dirtyField = true;
    if (toSolve) app.dirtySolve = app.dirtyField = true;
    if (toField) app.dirtyField = true;
    app.dirtyTitle = true;
}

static void refresh() {
    Params& p = app.p;
    if (app.dirtyMesh) { buildMesh(p, app.F.mesh); app.dirtyMesh = false; }
    if (app.dirtyAssemble) { assemble(p, app.F); app.dirtyAssemble = false; }
    if (app.dirtySolve) { solveLoad(p, app.F); recoverStress(p, app.F); app.dirtySolve = false; }
    if (app.dirtyField) { if (app.F.sxx.size() == app.F.mesh.X.size()) updateField(p, app.F); app.dirtyField = false; }
}

static void printHelp() {
    std::puts(
        "FEM sheet controls\n"
        "  Left mouse (click/drag)  move the load point\n"
        "  Q / E                    rotate load direction by -15 / +15 degrees\n"
        "  Left / Right arrow       decrease / increase load magnitude\n"
        "  Up / Down arrow          increase / decrease sheet thickness\n"
        "  , / .                    shrink / grow the load patch radius\n"
        "  1..6                     von Mises, sigma_xx, sigma_yy, tau_xy, max principal, min principal\n"
        "  [ / ]                    deformation exaggeration down / up\n"
        "  - / =                    coarser / finer mesh\n"
        "  H                        toggle hole       B  cycle supports       S  plane stress / strain\n"
        "  C                        smooth / banded contours     K  clip colour range\n"
        "  M                        mesh lines        U  undeformed outline   P  save shot.ppm\n"
        "  Esc                      quit\n"
        "Options: --thickness t --force F --angle deg --load x y --field 1..6 --banded --nohole --mesh 0..3 --shot f.ppm --selftest");
}

int main(int argc, char** argv) {
    bool shot = false;
    const char* shotPath = "shot.ppm";
    for (int i = 1; i < argc; ++i) {
        auto has = [&](int n) { return i + n < argc; };
        if (!std::strcmp(argv[i], "--selftest")) return selftest();
        else if (!std::strcmp(argv[i], "--shot")) { shot = true; if (has(1)) shotPath = argv[++i]; }
        else if (!std::strcmp(argv[i], "--thickness") && has(1)) app.p.thickness = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--force") && has(1)) app.p.Fmag = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--angle") && has(1)) app.p.angleDeg = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--load") && has(2)) { app.p.lx = std::atof(argv[i + 1]); app.p.ly = std::atof(argv[i + 2]); i += 2; }
        else if (!std::strcmp(argv[i], "--field") && has(1)) app.p.field = std::max(0, std::min(NFIELDS - 1, std::atoi(argv[++i]) - 1));
        else if (!std::strcmp(argv[i], "--banded")) { app.p.banded = true; }
        else if (!std::strcmp(argv[i], "--nohole")) { app.p.hole = false; }
        else if (!std::strcmp(argv[i], "--mesh") && has(1)) app.p.meshLevel = std::max(0, std::min(3, std::atoi(argv[++i])));
    }
    if (!glfwInit()) { std::puts("GLFW init failed"); return 1; }
    if (shot) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(app.ww, app.wh, "FEM sheet", nullptr, nullptr);
    if (!win) { std::puts("could not create window (is an OpenGL driver available?)"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetMouseButtonCallback(win, onMouse);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetKeyCallback(win, onKey);
    printHelp();

    app.p.nx = MESH_LEVELS[app.p.meshLevel][0];
    app.p.ny = MESH_LEVELS[app.p.meshLevel][1];
    while (!glfwWindowShouldClose(win)) {
        if (app.dirtyTexture) { makeTexture(app.p.banded); app.dirtyTexture = false; }
        refresh();
        if (app.dirtyTitle) { updateTitle(win); app.dirtyTitle = false; }
        drawScene(win);
        if (shot) { saveShot(win, shotPath); break; }
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
} // EOP
