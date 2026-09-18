# Blaze Quick Reference for FEM / Solid Mechanics

Compiled from the (now-defunct) Bitbucket wiki, since Atlassian sunset
Bitbucket Wikis/Issues on 20 Aug 2026 and took Blaze's docs with it.
This covers the ~80% you'll actually touch for FEM: vectors, matrices,
printing, input, arithmetic, and solving K·x = F.

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

## Where this came from

Content reconstructed from cached/archived copies of these now-dead pages:
`Getting Started`, `Vectors` / `Vector Types`, `Matrix Types`,
`Vector/Matrix Operations`, `Matrix Operations` (Linear Systems section).
Since the live Bitbucket wiki is gone, treat this as a working reference
rather than the full manual — if you hit something not covered here
(adaptors, custom types, BLAS/LAPACK wrappers, serialization to files),
say what you're trying to do and I'll dig up the relevant archived page.
