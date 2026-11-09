// fem_sheet.cpp  --  "FEM Stress Lab"
// Interactive 2D finite element sheet: bilinear quadrilateral (Q4) elements, plane stress / plane strain,
// optional hole, selectable thickness, movable load, nodal stress smoothing and an OpenGL contour map.
//
// Visual layer: dark theme, built-in 5x7 pixel font (no font library needed), title banner, dimension
// lines, support hatching, load arrow, pulsing peak-stress marker, yield line on the colour bar,
// and info / safety-check / controls cards.
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
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
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
static const char* FIELD_SHORT[NFIELDS] = {"VON MISES", "SIGMA XX", "SIGMA YY", "TAU XY",
                                           "MAX PRINCIPAL", "MIN PRINCIPAL"};
static const char* FIELD_TITLES[NFIELDS] = {"VON MISES EQUIVALENT STRESS", "NORMAL STRESS  SIGMA-XX",
                                            "NORMAL STRESS  SIGMA-YY", "SHEAR STRESS  TAU-XY",
                                            "MAXIMUM PRINCIPAL STRESS", "MINIMUM PRINCIPAL STRESS"};
static const char* BC_NAMES[3] = {"left edge fixed", "left+right edges fixed", "bottom edge fixed"};
static const int MESH_LEVELS[4][2] = {{20, 10}, {40, 20}, {80, 40}, {120, 60}};
static const int CLIP_PCT[4] = {100, 50, 25, 10};

struct Params {
    double W = 200.0, H = 100.0;     // sheet width and height [mm]
    int meshLevel = 1;               // index into MESH_LEVELS
    int nx = 40, ny = 20;            // cells along x and y
    double thickness = 5.0;          // [mm]
    double E = 200e3, nu = 0.3;      // steel [MPa]
    double yield = 250.0;            // yield strength used for the safety check [MPa] (mild steel)
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
    std::vector<std::array<int, 2>> bedges;   // boundary edges (outer edge + hole edge), for outlines
};

void buildMesh(const Params& p, Mesh& m) {
    m.X.clear();
    m.quads.clear();
    m.bedges.clear();
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
    // An edge that belongs to exactly one element lies on the boundary.
    std::map<std::pair<int, int>, int> count;
    for (auto& q : m.quads)
        for (int k = 0; k < 4; ++k) {
            int a = q[k], b = q[(k + 1) % 4];
            count[{std::min(a, b), std::max(a, b)}]++;
        }
    for (auto& kv : count)
        if (kv.second == 1) m.bedges.push_back({kv.first.first, kv.first.second});
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
    double vmMax = 0;                           // peak von Mises over all nodes (for the safety check)
    int peakNode = 0;                           // node with the largest |field| value
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
    F.vmMax = 0;
    F.peakNode = 0;
    double bestAbs = -1;
    for (int n = 0; n < N; ++n) {
        double sx = F.sxx[n], sy = F.syy[n], t = F.txy[n];
        double sz = p.planeStress ? 0.0 : p.nu * (sx + sy);
        double R = std::sqrt(0.25 * (sx - sy) * (sx - sy) + t * t);
        double vm = std::sqrt(0.5 * ((sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) + (sz - sx) * (sz - sx)) + 3 * t * t);
        F.vmMax = std::max(F.vmMax, vm);
        double v = 0;
        switch (p.field) {
            case VON_MISES: v = vm; break;
            case SIGMA_XX: v = sx; break;
            case SIGMA_YY: v = sy; break;
            case TAU_XY: v = t; break;
            case PRINC_MAX: v = 0.5 * (sx + sy) + R; break;
            case PRINC_MIN: v = 0.5 * (sx + sy) - R; break;
        }
        F.field[n] = v;
        if (std::fabs(v) > bestAbs) { bestAbs = std::fabs(v); F.peakNode = n; }
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
// 6. Rendering helpers: colour map, pixel font, rectangles, cards
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

struct Col { float r, g, b, a; };
static const Col C_WHITE = {1.0f, 1.0f, 1.0f, 1.0f};
static const Col C_DIM = {0.62f, 0.69f, 0.80f, 1.0f};
static const Col C_ACCENT = {0.30f, 0.85f, 1.00f, 1.0f};
static const Col C_WARM = {1.00f, 0.62f, 0.25f, 1.0f};
static const Col C_KEY = {1.00f, 0.82f, 0.35f, 1.0f};
static const Col C_PANEL = {0.08f, 0.11f, 0.17f, 0.92f};
static const Col C_GREEN = {0.30f, 0.90f, 0.50f, 1.0f};
static const Col C_AMBER = {1.00f, 0.75f, 0.20f, 1.0f};
static const Col C_RED = {1.00f, 0.32f, 0.32f, 1.0f};

static void setCol(const Col& c) { glColor4f(c.r, c.g, c.b, c.a); }

static void fillRect(double x, double y, double w, double h, const Col& c) {
    setCol(c);
    glBegin(GL_QUADS);
    glVertex2d(x, y); glVertex2d(x + w, y); glVertex2d(x + w, y + h); glVertex2d(x, y + h);
    glEnd();
}
static void strokeRect(double x, double y, double w, double h, const Col& c, float lw = 1.0f) {
    setCol(c); glLineWidth(lw);
    glBegin(GL_LINE_LOOP);
    glVertex2d(x, y); glVertex2d(x + w, y); glVertex2d(x + w, y + h); glVertex2d(x, y + h);
    glEnd();
}

// Classic 5x7 pixel font, ASCII 32..126. Each glyph = 5 columns, bit 0 = top row.
static const unsigned char FONT[95][5] = {
    {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},{0x14,0x7F,0x14,0x7F,0x14},
    {0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},
    {0x00,0x1C,0x22,0x41,0x00},{0x00,0x41,0x22,0x1C,0x00},{0x08,0x2A,0x1C,0x2A,0x08},{0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},{0x20,0x10,0x08,0x04,0x02},
    {0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},
    {0x18,0x14,0x12,0x7F,0x10},{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},{0x00,0x56,0x36,0x00,0x00},
    {0x00,0x08,0x14,0x22,0x41},{0x14,0x14,0x14,0x14,0x14},{0x41,0x22,0x14,0x08,0x00},{0x02,0x01,0x51,0x09,0x06},
    {0x32,0x49,0x79,0x41,0x3E},{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x01,0x01},{0x3E,0x41,0x41,0x51,0x32},
    {0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},
    {0x7F,0x40,0x40,0x40,0x40},{0x7F,0x02,0x04,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},{0x46,0x49,0x49,0x49,0x31},
    {0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},{0x1F,0x20,0x40,0x20,0x1F},{0x7F,0x20,0x18,0x20,0x7F},
    {0x63,0x14,0x08,0x14,0x63},{0x03,0x04,0x78,0x04,0x03},{0x61,0x51,0x49,0x45,0x43},{0x00,0x7F,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20},{0x00,0x41,0x41,0x7F,0x00},{0x04,0x02,0x01,0x02,0x04},{0x40,0x40,0x40,0x40,0x40},
    {0x00,0x01,0x02,0x04,0x00},{0x20,0x54,0x54,0x54,0x78},{0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},
    {0x38,0x44,0x44,0x48,0x7F},{0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x08,0x14,0x54,0x54,0x3C},
    {0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},{0x00,0x7F,0x10,0x28,0x44},
    {0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},{0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},
    {0x7C,0x14,0x14,0x14,0x08},{0x08,0x14,0x14,0x18,0x7C},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},{0x3C,0x40,0x30,0x40,0x3C},
    {0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},{0x44,0x64,0x54,0x4C,0x44},{0x00,0x08,0x36,0x41,0x00},
    {0x00,0x00,0x7F,0x00,0x00},{0x00,0x41,0x36,0x08,0x00},{0x02,0x01,0x02,0x04,0x02}};

// Width in pixels of a string drawn at integer scale sc (each font pixel becomes an sc x sc square).
static double textW(const char* s, int sc) { return (double)std::strlen(s) * 6.0 * sc - sc; }

// Draw text with its bottom-left corner at (x, y). If c2 is given the colour fades from c1 to c2.
static void text(const char* s, double x, double y, int sc, const Col& c1, const Col* c2 = nullptr) {
    const int n = (int)std::strlen(s);
    glBegin(GL_QUADS);
    for (int i = 0; i < n; ++i) {
        if (c2) {
            float t = n > 1 ? (float)i / (n - 1) : 0.f;
            glColor4f(c1.r + (c2->r - c1.r) * t, c1.g + (c2->g - c1.g) * t, c1.b + (c2->b - c1.b) * t, c1.a);
        } else {
            setCol(c1);
        }
        unsigned char ch = (unsigned char)s[i];
        if (ch >= 32 && ch < 127) {
            const unsigned char* gl = FONT[ch - 32];
            for (int col = 0; col < 5; ++col)
                for (int row = 0; row < 7; ++row)
                    if (gl[col] & (1 << row)) {
                        double px = x + (i * 6 + col) * sc, py = y + (6 - row) * sc;
                        glVertex2d(px, py); glVertex2d(px + sc, py);
                        glVertex2d(px + sc, py + sc); glVertex2d(px, py + sc);
                    }
        }
    }
    glEnd();
}
static void textShadow(const char* s, double x, double y, int sc, const Col& c1, const Col* c2 = nullptr) {
    Col sh = {0, 0, 0, 0.55f};
    text(s, x + sc, y - sc, sc, sh);
    text(s, x, y, sc, c1, c2);
}
static void textRight(const char* s, double xr, double y, int sc, const Col& c) { text(s, xr - textW(s, sc), y, sc, c); }
static void textCenter(const char* s, double xc, double y, int sc, const Col& c) { text(s, xc - textW(s, sc) / 2, y, sc, c); }

// Text on a small dark label background.
static void tag(const char* s, double x, double y, int sc, const Col& fg, const Col& bg) {
    fillRect(x - 5, y - 4, textW(s, sc) + 10, 7.0 * sc + 8, bg);
    text(s, x, y, sc, fg);
}

// A dashboard card: drop shadow, dark panel, accent line, title.
static void card(double x, double y, double w, double h, const char* title) {
    fillRect(x + 5, y - 5, w, h, Col{0, 0, 0, 0.35f});
    fillRect(x, y, w, h, C_PANEL);
    fillRect(x, y + h - 3, w, 3, Col{C_ACCENT.r, C_ACCENT.g, C_ACCENT.b, 0.85f});
    strokeRect(x + 0.5, y + 0.5, w - 1, h - 1, Col{1, 1, 1, 0.10f});
    text(title, x + 16, y + h - 30, 2, C_ACCENT);
}

// Nice number formatting for tick labels.
static void fmtVal(double v, char* buf, size_t n) {
    double a = std::fabs(v);
    if (a < 1000) std::snprintf(buf, n, "%.1f", v);
    else if (a < 1e5) std::snprintf(buf, n, "%.0f", v);
    else std::snprintf(buf, n, "%.2g", v);
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
    bool wantShot = false;
    int ww = 1280, wh = 800;
} app;

static const double CARD_Y = 20, CARD_H = 190;   // bottom dashboard row

struct View { double s, ox, oy; };
static View viewFor(int ww, int wh) {
    const double left = 100, right = 170, top = 110, bottom = 225;   // space reserved for UI around the sheet
    double availW = ww - left - right, availH = wh - top - bottom;
    View v;
    v.s = std::max(0.5, std::min(availW / app.p.W, availH / (app.p.H * 1.4)));
    v.ox = left + (availW - v.s * app.p.W) / 2;
    v.oy = bottom + (availH - v.s * app.p.H) / 2;
    return v;
}

static double clipTop(const Params& p, const Fem& F) {
    return F.fmin + (F.fmax - F.fmin) * (CLIP_PCT[p.clipIdx] / 100.0);
}

static void drawBackground(int ww, int wh) {
    glBegin(GL_QUADS);
    glColor3f(0.12f, 0.15f, 0.23f); glVertex2d(0, 0); glVertex2d(ww, 0);
    glColor3f(0.04f, 0.055f, 0.09f); glVertex2d(ww, wh); glVertex2d(0, wh);
    glEnd();
    glColor4f(1, 1, 1, 0.07f); glPointSize(2.0f);   // faint "engineering paper" dot grid
    glBegin(GL_POINTS);
    for (int x = 20; x < ww; x += 30)
        for (int y = 20; y < wh; y += 30) glVertex2d(x, y);
    glEnd();
}

static void drawHeader(int ww, int wh, const Params& p) {
    Col a = {0.35f, 0.85f, 1.0f, 1.0f}, b = {1.0f, 0.60f, 0.25f, 1.0f};
    textShadow("Finite Element Analysis", 30, wh - 62, 5, a, &b);
    text("Bilinear Q4 finite element sheet  -  live stress contours", 32, wh - 88, 2, C_DIM);
    textRight(FIELD_TITLES[p.field], ww - 30, wh - 50, 2, C_WHITE);
    textRight("press 1-6 to change the field", ww - 30, wh - 72, 1, C_DIM);
    const int SEG = 96;                              // rainbow strip, a preview of the colour map
    double x0 = 30, x1 = ww - 30;
    glBegin(GL_QUADS);
    for (int i = 0; i < SEG; ++i) {
        float r, g, bl;
        double xa = x0 + (x1 - x0) * i / SEG, xb = x0 + (x1 - x0) * (i + 1) / SEG;
        colormap((i + 0.5) / SEG, r, g, bl);
        glColor4f(r, g, bl, 0.95f);
        glVertex2d(xa, wh - 104); glVertex2d(xb, wh - 104); glVertex2d(xb, wh - 100); glVertex2d(xa, wh - 100);
    }
    glEnd();
}

static void drawColorbar(int ww, int wh, const Params& p, const Fem& F, double top) {
    double bx = ww - 135, bw = 26, by0 = 250, by1 = wh - 160;
    fillRect(bx + 4, by0 - 4, bw, by1 - by0, Col{0, 0, 0, 0.35f});
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, gTex);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord1f(0.5f / 256); glVertex2d(bx, by0); glVertex2d(bx + bw, by0);
    glTexCoord1f(255.5f / 256); glVertex2d(bx + bw, by1); glVertex2d(bx, by1);
    glEnd();
    glDisable(GL_TEXTURE_1D);
    strokeRect(bx, by0, bw, by1 - by0, Col{1, 1, 1, 0.6f}, 1.5f);

    for (int i = 0; i <= 5; ++i) {
        double t = i / 5.0, y = by0 + t * (by1 - by0);
        setCol(Col{1, 1, 1, 0.7f}); glLineWidth(1.0f);
        glBegin(GL_LINES); glVertex2d(bx + bw, y); glVertex2d(bx + bw + 6, y); glEnd();
        char buf[32];
        fmtVal(F.fmin + t * (top - F.fmin), buf, sizeof buf);
        text(buf, bx + bw + 10, y - 7, 2, C_WHITE);
    }
    text(FIELD_SHORT[p.field], bx, by1 + 38, 1, C_DIM);
    text("[MPa]", bx, by1 + 16, 2, C_WHITE);
    if (p.clipIdx > 0) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "CLIPPED TO %d%%", CLIP_PCT[p.clipIdx]);
        text(buf, bx - 24, by0 - 22, 1, C_WARM);
    }
    // yield line (only meaningful for von Mises)
    if (p.field == VON_MISES && p.yield > F.fmin && p.yield < top) {
        double y = by0 + (p.yield - F.fmin) / (top - F.fmin) * (by1 - by0);
        setCol(C_WHITE); glLineWidth(3.0f);
        glBegin(GL_LINES); glVertex2d(bx - 8, y); glVertex2d(bx + bw + 8, y); glEnd();
        textRight("YIELD", bx - 12, y - 3, 1, C_WHITE);
    }
}

static void drawInfoCard(const Params& p, const Fem& F) {
    double x = 30, w = 430, y = CARD_Y, h = CARD_H;
    card(x, y, w, h, "MODEL");
    double ty = y + h - 60;
    char b[96], c[64];
    auto row = [&](const char* k, const char* val) {
        text(k, x + 16, ty, 2, C_DIM);
        text(val, x + 16 + 7 * 12, ty, 2, C_WHITE);
        ty -= 18;
    };
    std::snprintf(b, sizeof b, "%.0f x %.0f x %.1f mm", p.W, p.H, p.thickness); row("Sheet", b);
    std::snprintf(b, sizeof b, "E=%.0f GPa  nu=%.2f", p.E / 1000.0, p.nu); row("Steel", b);
    std::snprintf(b, sizeof b, "%.0f N @ %.0f deg", p.Fmag, p.angleDeg); row("Load", b);
    if (p.hole) std::snprintf(c, sizeof c, "hole R=%.0f mm", p.holeR); else std::snprintf(c, sizeof c, "no hole");
    std::snprintf(b, sizeof b, "%s, %s", p.planeStress ? "Plane stress" : "Plane strain", c); row("Mode", b);
    std::snprintf(b, sizeof b, "%d x %d  (%zu nodes)", p.nx, p.ny, F.mesh.X.size()); row("Mesh", b);
    row("Fixed", BC_NAMES[p.bcMode]);
    double defScale = F.umax > 0 ? 0.08 * p.H / F.umax * p.defMult : 0.0;
    if (F.umax >= 1e-3) std::snprintf(c, sizeof c, "%.4f mm", F.umax); else std::snprintf(c, sizeof c, "%.2e mm", F.umax);
    std::snprintf(b, sizeof b, "%s  (def x%.0f)", c, defScale); row("Disp", b);
}

static void drawResultCard(const Params& p, const Fem& F) {
    double x = 490, w = 300, y = CARD_Y, h = CARD_H, T = y + h;
    card(x, y, w, h, "SAFETY CHECK");
    char b[64];
    text("PEAK VON MISES STRESS", x + 16, T - 50, 1, C_DIM);
    if (F.vmMax < 1e4) std::snprintf(b, sizeof b, "%.1f", F.vmMax); else std::snprintf(b, sizeof b, "%.3g", F.vmMax);
    text(b, x + 16, T - 86, 4, C_WHITE);
    text("MPa", x + 16 + textW(b, 4) + 10, T - 86, 2, C_DIM);
    std::snprintf(b, sizeof b, "Yield strength %.0f MPa", p.yield);
    text(b, x + 16, T - 112, 1, C_DIM);

    double sf = F.vmMax > 1e-9 ? p.yield / F.vmMax : 999.0;
    Col sc = sf >= 1.5 ? C_GREEN : sf >= 1.0 ? C_AMBER : C_RED;
    const char* verdict = sf >= 1.5 ? "SAFE" : sf >= 1.0 ? "MARGINAL" : "YIELDS";
    text("SAFETY FACTOR", x + 16, T - 136, 1, C_DIM);
    if (sf > 99) std::snprintf(b, sizeof b, ">99"); else std::snprintf(b, sizeof b, "%.2f", sf);
    text(b, x + 16, T - 168, 3, sc);
    fillRect(x + 140, T - 172, 144, 30, Col{sc.r, sc.g, sc.b, 0.22f});
    strokeRect(x + 140, T - 172, 144, 30, sc, 1.5f);
    textCenter(verdict, x + 140 + 72, T - 163, 2, sc);
}

static void drawControlsCard(int ww) {
    double x = 810, w = ww - 30 - x, y = CARD_Y, h = CARD_H, T = y + h;
    card(x, y, w, h, "CONTROLS");
    static const char* L[][2] = {
        {"Mouse", "move load point"}, {"Q / E", "rotate load 15 deg"}, {"< / >", "load magnitude"},
        {"Up/Down", "sheet thickness"}, {", / .", "load patch size"}, {"1 - 6", "stress field"},
        {"[ / ]", "deformation scale"}, {"- / =", "mesh density"}, {"H", "toggle hole"}};
    static const char* R[][2] = {
        {"B", "cycle supports"}, {"S", "stress / strain"}, {"C", "smooth / banded"},
        {"K", "clip colour range"}, {"M", "mesh lines"}, {"U", "undeformed outline"},
        {"P", "save shot.ppm"}, {"Esc", "quit"}};
    for (int i = 0; i < 9; ++i) {
        double ty = T - 56 - i * 15;
        text(L[i][0], x + 16, ty, 1, C_KEY);
        text(L[i][1], x + 16 + 54, ty, 1, C_DIM);
    }
    for (int i = 0; i < 8; ++i) {
        double ty = T - 56 - i * 15;
        text(R[i][0], x + 16 + 205, ty, 1, C_KEY);
        text(R[i][1], x + 16 + 205 + 30, ty, 1, C_DIM);
    }
}

static void drawScene(GLFWwindow* win) {
    int ww, wh, fw, fh;
    glfwGetWindowSize(win, &ww, &wh);
    glfwGetFramebufferSize(win, &fw, &fh);
    app.ww = ww; app.wh = wh;
    glViewport(0, 0, fw, fh);
    glClearColor(0.05f, 0.065f, 0.10f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, ww, 0, wh, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);

    const Params& p = app.p;
    const Fem& F = app.F;
    drawBackground(ww, wh);
    drawHeader(ww, wh, p);
    View v = viewFor(ww, wh);
    const int N = (int)F.mesh.X.size();
    if (N == 0 || (int)F.u.size() != 2 * N || (int)F.field.size() != N) return;
    const double tt = glfwGetTime();

    double autoScale = F.umax > 0 ? 0.08 * p.H / F.umax : 0.0;
    double def = autoScale * p.defMult;
    std::vector<Vector2d> P(N), P0(N);
    for (int n = 0; n < N; ++n) {
        Vector2d x0 = F.mesh.X[n];
        Vector2d x1 = x0 + def * Vector2d(F.u(2 * n), F.u(2 * n + 1));
        P0[n] = Vector2d(v.ox + v.s * x0.x(), v.oy + v.s * x0.y());
        P[n] = Vector2d(v.ox + v.s * x1.x(), v.oy + v.s * x1.y());
    }
    const double sx0 = v.ox, sx1 = v.ox + v.s * p.W, sy0 = v.oy, sy1 = v.oy + v.s * p.H;

    // ---- dimension lines (undeformed geometry) ------------------------------------------------
    {
        Col dc = {0.60f, 0.72f, 0.90f, 0.85f}, bg = {0.06f, 0.08f, 0.13f, 1.0f};
        char b[32];
        double yd = sy1 + 26, xd = sx0 - 54;
        setCol(dc); glLineWidth(1.0f);
        glBegin(GL_LINES);
        glVertex2d(sx0, yd); glVertex2d(sx1, yd);
        glVertex2d(sx0, yd - 6); glVertex2d(sx0, yd + 6); glVertex2d(sx1, yd - 6); glVertex2d(sx1, yd + 6);
        glVertex2d(xd, sy0); glVertex2d(xd, sy1);
        glVertex2d(xd - 6, sy0); glVertex2d(xd + 6, sy0); glVertex2d(xd - 6, sy1); glVertex2d(xd + 6, sy1);
        glEnd();
        std::snprintf(b, sizeof b, "%.0f mm", p.W);
        tag(b, (sx0 + sx1) / 2 - textW(b, 2) / 2, yd - 7, 2, C_DIM, bg);
        std::snprintf(b, sizeof b, "%.0f mm", p.H);
        tag(b, xd - textW(b, 2) / 2, (sy0 + sy1) / 2 - 7, 2, C_DIM, bg);
    }

    // ---- drop shadow under the sheet ---------------------------------------------------------
    glColor4f(0, 0, 0, 0.38f);
    glBegin(GL_TRIANGLES);
    for (auto& q : F.mesh.quads) {
        int idx[6] = {0, 1, 2, 0, 2, 3};
        for (int k : idx) glVertex2d(P[q[k]].x() + 7, P[q[k]].y() - 7);
    }
    glEnd();

    // ---- contour map: scalar -> 1D texture coordinate ----------------------------------------
    double top = clipTop(p, F);
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

    if (p.showMesh) {
        glColor4f(0, 0, 0, 0.12f); glLineWidth(1.0f);
        for (auto& q : F.mesh.quads) {
            glBegin(GL_LINE_LOOP);
            for (int k = 0; k < 4; ++k) glVertex2d(P[q[k]].x(), P[q[k]].y());
            glEnd();
        }
    }
    if (p.showUndeformed) {   // dashed ghost of the original shape
        glEnable(GL_LINE_STIPPLE); glLineStipple(3, 0x5555);
        glColor4f(1, 1, 1, 0.60f); glLineWidth(1.2f);
        glBegin(GL_LINES);
        for (auto& e : F.mesh.bedges) { glVertex2d(P0[e[0]].x(), P0[e[0]].y()); glVertex2d(P0[e[1]].x(), P0[e[1]].y()); }
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }
    glColor4f(1, 1, 1, 0.95f); glLineWidth(2.2f);   // crisp outline of the deformed sheet
    glBegin(GL_LINES);
    for (auto& e : F.mesh.bedges) { glVertex2d(P[e[0]].x(), P[e[0]].y()); glVertex2d(P[e[1]].x(), P[e[1]].y()); }
    glEnd();

    // ---- supports: engineering-style hatching ------------------------------------------------
    glColor4f(0.5f, 0.85f, 1.0f, 0.95f); glLineWidth(1.6f);
    glBegin(GL_LINES);
    for (int n = 0; n < N; ++n)
        if (F.fixedDof[2 * n]) {
            double ox = 0, oy = 0;
            if (p.bcMode == 2) oy = -1;
            else if (F.mesh.X[n].x() < p.W / 2) ox = -1;
            else ox = 1;
            Vector2d a = P0[n];
            Vector2d b = a + 10.0 * Vector2d(ox, oy) + (ox != 0 ? Vector2d(0, -10) : Vector2d(-10, 0));
            glVertex2d(a.x(), a.y()); glVertex2d(b.x(), b.y());
        }
    glEnd();

    // ---- hole label --------------------------------------------------------------------------
    if (p.hole) {
        Vector2d acc(0, 0); int cnt = 0;
        double rr = p.holeR + 1.5 * std::max(p.W / p.nx, p.H / p.ny);
        for (int n = 0; n < N; ++n)
            if ((F.mesh.X[n] - Vector2d(p.W / 2, p.H / 2)).norm() < rr) { acc += P[n]; ++cnt; }
        if (cnt) {
            acc /= (double)cnt;
            char b[32];
            std::snprintf(b, sizeof b, "R = %.0f mm", p.holeR);
            int sc = v.s * p.holeR * 2 > 120 ? 2 : 1;
            textCenter(b, acc.x(), acc.y() - 3.5 * sc, sc, C_DIM);
        }
    }

    // ---- load: patch disc, arrow and label ---------------------------------------------------
    {
        double cx = v.ox + v.s * p.lx, cy = v.oy + v.s * p.ly;
        double a = p.angleDeg * M_PI / 180.0, dxn = std::cos(a), dyn = std::sin(a);
        if (!p.loadEdge) {
            double pr = v.s * p.loadRadius;
            glColor4f(C_WARM.r, C_WARM.g, C_WARM.b, 0.18f);
            glBegin(GL_TRIANGLE_FAN);
            glVertex2d(cx, cy);
            for (int i = 0; i <= 40; ++i) glVertex2d(cx + pr * std::cos(i * M_PI / 20), cy + pr * std::sin(i * M_PI / 20));
            glEnd();
            glColor4f(C_WARM.r, C_WARM.g, C_WARM.b, 0.65f + 0.35f * (float)std::sin(tt * 3.0)); glLineWidth(2.0f);
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < 40; ++i) glVertex2d(cx + pr * std::cos(i * M_PI / 20), cy + pr * std::sin(i * M_PI / 20));
            glEnd();

            double len = 80;
            double tx = cx - dxn * len, ty = cy - dyn * len;
            glColor4f(0, 0, 0, 0.55f); glLineWidth(6.0f);
            glBegin(GL_LINES); glVertex2d(tx, ty); glVertex2d(cx - dxn * 10, cy - dyn * 10); glEnd();
            setCol(C_WARM); glLineWidth(3.0f);
            glBegin(GL_LINES); glVertex2d(tx, ty); glVertex2d(cx - dxn * 10, cy - dyn * 10); glEnd();
            glBegin(GL_TRIANGLES);   // arrow head
            glVertex2d(cx, cy);
            glVertex2d(cx - dxn * 20 + dyn * 8, cy - dyn * 20 - dxn * 8);
            glVertex2d(cx - dxn * 20 - dyn * 8, cy - dyn * 20 + dxn * 8);
            glEnd();

            char b[48];
            std::snprintf(b, sizeof b, "F = %.0f N", p.Fmag);
            double w = textW(b, 2);
            double lxp = tx - dxn * (w / 2 + 10) - w / 2, lyp = ty - dyn * 16 - 7;
            lxp = std::max(8.0, std::min(ww - w - 8.0, lxp));
            lyp = std::max(8.0, std::min(wh - 30.0, lyp));
            tag(b, lxp, lyp, 2, C_WARM, Col{0, 0, 0, 0.6f});
        }
    }

    // ---- peak stress marker (pulsing ring + label) -------------------------------------------
    {
        Vector2d pk = P[F.peakNode];
        double r = 10 + 3 * std::sin(tt * 4.0);
        glColor4f(0, 0, 0, 0.7f); glLineWidth(5.0f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 32; ++i) glVertex2d(pk.x() + r * std::cos(i * M_PI / 16), pk.y() + r * std::sin(i * M_PI / 16));
        glEnd();
        glColor4f(1, 1, 1, 1); glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 32; ++i) glVertex2d(pk.x() + r * std::cos(i * M_PI / 16), pk.y() + r * std::sin(i * M_PI / 16));
        glEnd();
        char vb[32], b[64];
        fmtVal(F.field[F.peakNode], vb, sizeof vb);
        std::snprintf(b, sizeof b, "PEAK %s MPa", vb);
        double w = textW(b, 2);
        double lxp = std::max(8.0, std::min(ww - w - 8.0, pk.x() + 20));
        double lyp = std::max(8.0, std::min(wh - 30.0, pk.y() + 18));
        tag(b, lxp, lyp, 2, C_WHITE, Col{0, 0, 0, 0.7f});
    }

    drawColorbar(ww, wh, p, F, top);
    drawInfoCard(p, F);
    drawResultCard(p, F);
    drawControlsCard(ww);
    glLineWidth(1.0f);
}

static void updateTitle(GLFWwindow* win) {
    const Params& p = app.p;
    char buf[160];
    std::snprintf(buf, sizeof buf, "FEM Stress Lab  |  %s  |  %s", FIELD_NAMES[p.field],
                  p.planeStress ? "plane stress" : "plane strain");
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
        case GLFW_KEY_P: app.wantShot = true; break;   // saved in the main loop, right after drawing
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
        "FEM Stress Lab controls\n"
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
    GLFWwindow* win = glfwCreateWindow(app.ww, app.wh, "FEM Stress Lab", nullptr, nullptr);
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
        if (app.wantShot) { saveShot(win, "shot.ppm"); app.wantShot = false; }
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
