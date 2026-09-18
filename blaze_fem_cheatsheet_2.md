# Blaze Quick Reference for FEM / Solid Mechanics

Compiled from the (now-defunct) Bitbucket wiki, since Atlassian sunset
Bitbucket Wikis/Issues on 20 Aug 2026 and took Blaze's docs with it.
This covers the ~80% you'll actually touch for FEM: vectors, matrices,
printing, input, arithmetic, and solving K·x = F.

Rule of thumb for Blaze in general: auto on a row()/column()/submatrix()/etc. call always gives you a view, never a copy. If you want a copy, name the concrete type explicitly.

```cpp
#include <blaze/Math.h>   // pulls in the whole math module
```
Everything lives in namespace `blaze`.

---

## 1. Vector Types

```cpp
using blaze::StaticVector;
using blaze::DynamicVector;
using blaze::columnVector;
using blaze::rowVector;

// Fixed-size (size known at compile time) — good for e.g. 3-DOF nodal displacement
StaticVector<double,3UL> a{ 4.0, -2.0, 5.0 };

// Runtime-sized — good for global displacement/force vectors
DynamicVector<double> b( 3UL );   // size 3, uninitialized
b[0] = 2.0; b[1] = 5.0; b[2] = -3.0;

// Default orientation is column vector unless stated otherwise
DynamicVector<int,rowVector> r{ 4, 5, 6 };
```

There's also `CompressedVector` (sparse) and `HybridVector` (stack storage
up to a max size, dynamic length) — rarely needed for typical FEM assembly
unless you're doing sparse global systems.

---

## 2. Matrix Types

```cpp
using blaze::StaticMatrix;
using blaze::DynamicMatrix;
using blaze::rowMajor;
using blaze::columnMajor;

StaticMatrix<double,3UL,3UL> Ke;      // e.g. a 3x3 element stiffness block
DynamicMatrix<double> K( n, n, 0.0 ); // n x n global stiffness, zero-initialized
DynamicMatrix<double,columnMajor> Kc( n, n );  // column-major (matters for LAPACK calls)
```

`CompressedMatrix<double>` is the sparse equivalent — worth switching to
once your global stiffness matrix gets big, since FEM matrices are
typically very sparse.

---

## 3. Construction & Element Access

```cpp
DynamicMatrix<double> K( 4, 4 );   // rows, cols
K(0,0) = 12.0;
K(0,1) = -6.0;
// ...

DynamicVector<double> F( 4, 0.0 ); // size 4, all zeros
F[2] = 100.0;   // applied nodal load
```

Resizing:
```cpp
K.resize( 6, 6 );      // does NOT preserve values by default
K.resize( 6, 6, true ); // preserve() = true keeps old values, pads with 0
```

---

## 4. Printing ("the cheese")

Blaze overloads `operator<<` for every vector/matrix type — this is the
big convenience:

```cpp
#include <iostream>
std::cout << "K =\n" << K << "\n";
std::cout << "F =\n" << F << "\n";
```

Output is auto-formatted with aligned columns, e.g.:
```
( 12  -6   0   0 )
( -6  12  -6   0 )
(  0  -6  12  -6 )
(  0   0  -6  12 )
```

There's no special "pretty-print with N decimals" flag built in — for that,
just set `std::cout` precision/format flags beforehand:
```cpp
std::cout << std::fixed << std::setprecision(4) << K << "\n";
```

---

## 5. Reading User Input

Blaze doesn't have a magic "type into a matrix" function — you just loop
with `std::cin` and use element access, same as any C++ container:

```cpp
DynamicMatrix<double> K( n, n );
for( size_t i = 0UL; i < n; ++i )
   for( size_t j = 0UL; j < n; ++j )
      std::cin >> K(i,j);

DynamicVector<double> F( n );
for( size_t i = 0UL; i < n; ++i )
   std::cin >> F[i];
```

(The `operator>>` you'll see referenced in Blaze's docs is for
**binary serialization archives** — reading/writing vectors/matrices
to/from files, not console input — see §7 below.)

---

## 6. Core Arithmetic

```cpp
DynamicVector<double> u = a + b;      // vector addition
DynamicVector<double> d = a - b;      // subtraction
DynamicVector<double> s = 2.0 * a;    // scalar multiplication

DynamicMatrix<double> C = A + B;      // matrix addition
DynamicMatrix<double> Ct = trans(A);  // transpose

// Vector/vector
double dot = trans(a) * b;            // inner (dot) product
DynamicMatrix<double> outer = a * trans(b);  // outer product -> matrix

// Matrix/vector — the FEM workhorse: F = K * u
DynamicVector<double> F = K * u;

// Matrix/matrix
DynamicMatrix<double> Kglobal = A * B;
```

Dimension mismatches throw at runtime (not silently broadcast) — Blaze is
strict about shape checking, which is good for catching assembly bugs.

---

## 7. Solving K·x = F (the actual FEM payoff)

Two options:

**A. Direct solve (preferred, since Blaze 3.7+):**
```cpp
DynamicMatrix<double> K;   // system matrix, populated
DynamicVector<double> F;   // right-hand side
DynamicVector<double> x;   // solution (displacements)

solve( K, x, F );          // computes x
// or:
x = solve( K, F );         // equivalent, functional style
```
Blaze auto-picks: hand-optimized kernels for small (≤6x6) systems,
LAPACK for larger ones. This does **not** explicitly invert K — it's a
direct solver (LU-based under the hood), so it's both faster and more
numerically stable than inverting.

**B. Explicit inverse (avoid unless you actually need K⁻¹ itself):**
```cpp
DynamicVector<double> x = inv(K) * F;
```
Blaze is smart enough to recognize this specific pattern (`inv(A)*b`)
and route it through a solver rather than literally inverting — but
`solve()` is still the clearer, recommended form.

For symmetric stiffness matrices (typical in solid mechanics), wrap in
`SymmetricMatrix<DynamicMatrix<double>>` — Blaze will use symmetric-aware
kernels/LAPACK routines automatically.

---

## 8. A Few More Useful Bits

```cpp
// Views — work on part of a matrix/vector without copying
auto sub = submatrix( K, 0, 0, 3, 3 );   // 3x3 block starting at (0,0)
auto rowi = row( K, 2 );                 // row 2 as a view
auto coli = column( K, 2 );              // column 2 as a view

// Zero/identity helpers
DynamicMatrix<double> I( blaze::IdentityMatrix<double>( n ) );

// Norms — handy for convergence checks in nonlinear FEM
double residualNorm = blaze::norm( F - K*u );  // L2 norm
```

---

## 9. Views (zero-copy windows into vectors/matrices)

Very useful in FEM for pulling out DOF blocks, applying boundary
conditions, or working with element-local data inside a global matrix
without copying.

```cpp
using blaze::submatrix;
using blaze::subvector;
using blaze::row;
using blaze::column;
using blaze::band;

DynamicMatrix<double> K( 100, 100 );
DynamicVector<double> F( 100 );

// A view is a live alias — writing to it writes to K itself
auto Ke = submatrix( K, 10, 10, 6, 6 );   // 6x6 block starting at (10,10)
Ke += localStiffness;                      // scatter-add element matrix

auto Fe = subvector( F, 10, 6 );           // 6 entries starting at index 10
Fe += localForce;

auto r = row( K, 3 );        // 3rd row as a view
auto c = column( K, 3 );     // 3rd column as a view
auto diag = band( K, 0L );   // main diagonal as a view (0 = diagonal, +n/-n = off-diagonals)
```

Views support the same arithmetic as regular vectors/matrices, and can
sit on either side of `=` (read or write).

---

## 10. Symmetric & Triangular Matrix Adaptors

FEM stiffness matrices are (usually) symmetric — wrapping yours in
`SymmetricMatrix` gets you automatic symmetric storage/kernels and
frees you from manually mirroring `K(i,j) = K(j,i)`:

```cpp
using blaze::SymmetricMatrix;
using blaze::DynamicMatrix;

SymmetricMatrix< DynamicMatrix<double> > K( n );
K(2,5) = 3.7;          // automatically sets K(5,2) = 3.7 too
```

Triangular adaptors (`LowerMatrix`, `UpperMatrix`, `UniLowerMatrix`,
`UniUpperMatrix`) are handy after an LU-type factorization step, or when
you're implementing your own solver instead of calling `solve()`:

```cpp
using blaze::LowerMatrix;
LowerMatrix< DynamicMatrix<double> > L( n );
// Attempting to write above the diagonal throws std::invalid_argument
```

All of these are "adaptors" — thin wrappers that enforce the structural
constraint at write-time and pick faster algorithms at compute-time; they
are not separate storage formats you convert to/from manually.

---

## 11. Matrix Decomposition / Eigenvalues (for later — modal analysis etc.)

Beyond straight `solve()`, Blaze wraps LAPACK for the heavier stuff:

```cpp
#include <blaze/Math.h>

DynamicMatrix<double> A, L, U, P;   // LU decomposition
blaze::lu( A, L, U, P );            // A = P^T * L * U

DynamicMatrix<double> Q, R;         // QR decomposition
blaze::qr( A, Q, R );

DynamicVector<blaze::complex<double>> eigenvalues;
DynamicMatrix<blaze::complex<double>> eigenvectors;
blaze::eigen( A, eigenvalues, eigenvectors );
```
For symmetric matrices, `blaze::eigen()` on a `SymmetricMatrix` gives you
real eigenvalues directly — relevant if/when you get to natural-frequency
or buckling analysis on top of the static FEM solve.

---

## 12. Build & Configuration Notes

Blaze is **header-only** — there's no library to link against for the
core functionality:

```cpp
#include <blaze/Math.h>
```

Compile with a reasonably modern flag set and vectorization enabled:
```
g++ -std=c++17 -O3 -DNDEBUG -mavx -o sim main.cpp
```

Optional, only needed if you want BLAS/LAPACK-backed heavy lifting
(large dense solves, eigen decompositions) rather than Blaze's own
kernels — link against a BLAS/LAPACK implementation (OpenBLAS is the
common choice on Linux):
```
g++ -std=c++17 -O3 -DNDEBUG -mavx main.cpp -lopenblas -llapack -o sim
```
Without linking BLAS/LAPACK, Blaze still works fine for small-to-medium
FEM problems — it just falls back to its own kernels, which are
perfectly good for a 2-week learning project.

---

## Where this came from

Content reconstructed (paraphrased, not copy-pasted) from cached/archived
copies of the now-dead Bitbucket wiki pages: `Getting Started`, `Vectors`
/ `Vector Types`, `Matrix Types`, `Vector Operations`, `Matrix Operations`
(Linear Systems, Decomposition), `Views` (Submatrices, Subvectors, Bands),
and `Adaptors` (Symmetric/Triangular Matrices).

Two things worth knowing:
1. **This is a working reference, not the literal original text.** I
   can't reproduce large verbatim chunks of someone else's docs, so
   this is written in my own words from what I could dig up — the
   concepts and APIs are accurate, but if you need the *exact* original
   wording (e.g. for a citation), pull it yourself from a Wayback
   Machine snapshot of `bitbucket.org/blaze-lib/blaze/wiki/*`.
2. **Not every corner is covered.** Things I haven't included: sparse
   assembly patterns in depth, custom operations/custom data types,
   BLAS/LAPACK function-by-function reference, serialization to files,
   shared-memory parallelization (OpenMP/HPX/threads), the FAQ. If your
   FEM project needs any of those, tell me and I'll go dig up that
   specific page and add a section.
