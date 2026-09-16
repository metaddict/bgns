
# bgns: Biweight Graph and Network Statistics

`bgns` is an R package for robust biweight midcorrelation (`bicor`) and exact
bicor-based k-nearest-neighbor graph construction on large, high-dimensional
numeric matrices.

Biweight midcorrelation is a robust correlation measure that reduces the
influence of outlying observations. `bgns` combines this similarity measure
with a performance-oriented C++17 backend designed to reduce memory
requirements during correlation-edge and KNN construction.

The package is particularly suited to gene-expression, single-cell, metacell,
and other biological data in which coordinated variation is an informative
similarity signal, although the implementation is not restricted to biological
matrices.

## Why bgns?

Large biological matrices create two related computational problems:

1. robust pairwise similarity can be considerably more expensive than simple
   Euclidean or cosine distance; and
2. materializing a complete similarity matrix requires memory proportional to
   the square of the number of items being compared.

`bgns` addresses these problems with exact bicor computation combined with
streamed and panelized execution paths. For KNN construction, it maintains
bounded top-k candidate sets rather than requiring a complete dense similarity
matrix to be retained in memory.

The resulting neighbors are exact under the implemented bicor score.
`bicor_knn()` is not an approximate-nearest-neighbor index.

## What it provides

- Robust biweight midcorrelation for dense numeric matrices.
- Exact bicor-based KNN graph construction.
- Tidy correlation-edge output without requiring a complete dense similarity
  matrix in applicable workflows.
- Dense, sparse `Matrix`, and mixed dense/sparse input paths.
- Pairwise finite-overlap handling for missing values.
- Explicit `min_overlap` control.
- C++17 computational kernels.
- BLAS-backed dense computation.
- Panelized execution to bound temporary memory use.
- Adaptive sparse-overlap strategies, including bitset-based paths.
- Runtime-gated SIMD kernels when supported by the compiler target.
- Optional OpenMP parallelism.

The public R API is intentionally small:

- `bicor()`
- `bicor_knn()`

## Installation

Install the current version from GitHub:

```r
install.packages("remotes")
remotes::install_github("metaddict/bgns")
```

Then load the package:

```r
library(bgns)
```

## Quick start

Rows are observations. Columns are the items whose similarity is being
evaluated.

```r
set.seed(1)

x <- matrix(
  rnorm(400),
  nrow = 40,
  ncol = 10
)

x[1, 2] <- NA_real_
```

### Biweight correlation matrix

```r
cm <- bicor(x)

round(cm[1:5, 1:5], 3)
```

### Correlation edges

Instead of returning the complete correlation matrix, `bicor()` can stream
qualifying correlations into an edge table:

```r
edges <- bicor(
  x,
  tidy = TRUE,
  threshold = 0.2
)

head(edges)
```

This is useful when the downstream analysis operates directly on a similarity
graph rather than on the full pairwise matrix.

### Exact bicor KNN graph

```r
kn <- bicor_knn(
  x,
  knn = 3,
  threshold = 0
)

head(kn)
```

For every target item, `bicor_knn()` retains the exact highest-scoring
neighbors under the bicor similarity measure.

## Sparse matrices

Sparse `Matrix` inputs are supported in edge and KNN workflows.

```r
library(Matrix)

set.seed(2)

xs <- rsparsematrix(
  100,
  20,
  density = 0.15
)

edges_sparse <- bicor(
  xs,
  tidy = TRUE,
  threshold = 0.2
)

kn_sparse <- bicor_knn(
  xs,
  knn = 5,
  threshold = 0
)
```

The implementation contains dedicated sparse execution paths rather than
requiring users to convert sparse matrices to dense matrices first.

## Mixed dense and sparse inputs

`bicor_knn()` can also construct bipartite KNN relationships between two
matrices with the same observation dimension.

```r
set.seed(3)

y <- matrix(
  rnorm(100 * 6),
  nrow = 100,
  ncol = 6
)

colnames(xs) <- paste0("source_", seq_len(ncol(xs)))
colnames(y)  <- paste0("target_", seq_len(ncol(y)))

kn_xy <- bicor_knn(
  xs,
  y,
  knn = 3,
  bipartite_levels = "separate"
)

head(kn_xy)
```

For bipartite KNN, neighbors are selected from columns of `x` for each target
column of `y`.

## Missing values and undefined correlations

With

```r
pairwise.complete.obs = TRUE
```

each pair is evaluated using its finite overlap.

A pair must satisfy the requested `min_overlap`. Constant columns,
all-missing columns, or columns with otherwise undefined robust variance
produce undefined correlations.

For dense self-correlation output, such diagonal values remain `NA`; they are
not artificially replaced with `1`.

With

```r
pairwise.complete.obs = FALSE
```

columns containing incomplete observations are ineligible for the affected
comparisons.

## Exact KNN and memory behavior

Suppose a matrix contains `p` columns.

A conventional full similarity calculation may require storage for a
`p × p` similarity matrix. That storage becomes increasingly expensive as
`p` grows.

`bicor_knn()` instead performs exact comparisons while retaining bounded
top-k candidate sets and using panelized computation.

Conceptually:

```text
full similarity workflow

input
  |
  v
all pairwise similarities
  |
  v
p x p similarity matrix
  |
  v
select top-k


bgns KNN workflow

input
  |
  v
panelized bicor computation
  |
  v
bounded top-k candidates
  |
  v
exact KNN graph
```

This reduces the need to materialize the complete similarity matrix while
preserving exact KNN semantics.

## Robust similarity rather than geometric distance

Euclidean and cosine distances are often useful for nearest-neighbor
construction, but they describe different relationships from correlation.

`bgns` is intended for analyses in which coordinated variation is itself the
signal of interest.

Typical examples include:

- gene co-expression;
- transcriptional programs;
- single-cell or metacell state relationships;
- robust similarity graphs;
- high-dimensional biological measurements containing outlying observations.

The package generates similarity or KNN relationships. Downstream clustering,
community detection, visualization, or biological annotation can be performed
with other R packages.

## Runtime controls

Several environment variables control computational behavior.

### Memory budget

```text
BGNS_MEM_MB
```

Controls the scratch-memory budget used by panelized computation.

The default is 256 MB.

### Thread limit

```text
BGNS_NUM_THREADS
```

Controls the maximum number of OpenMP threads when OpenMP support is available.

The default is capped conservatively.

### SIMD

```text
BGNS_SIMD
```

Set to `0`, `false`, or `off` to disable compiled SIMD kernels.

Runtime SIMD paths are used only when supported by the compiled target and
runtime environment.

### Sparse bitset strategy

```text
BGNS_BITSET_BETA
```

Controls adaptive bitset construction used by sparse-overlap paths.

For ordinary use, exported R arguments such as `min_overlap` should be
preferred over modifying internal environment controls.

## Validation

The numerical core of version 0.4.2 has been tested against independent R
reference calculations across:

- dense correlation;
- rectangular `x`/`y` correlation;
- missing-value handling;
- sparse and dense parity;
- mixed dense/sparse inputs;
- exact KNN selection;
- sparse KNN with missing observations;
- thread-consistency cases; and
- randomized reference cases.

The source tree also contains package-level unit tests covering API behavior
and release fixes.

## Planned examples and benchmarks

Reproducible examples will be used to illustrate three complementary aspects
of the package:

1. robustness of bicor relative to Pearson correlation under increasing
   outlier contamination;
2. computational and memory behavior of exact KNN construction as matrix size
   increases; and
3. construction of biologically interpretable similarity graphs from
   expression data.

Benchmark figures should be interpreted together with their input dimensions,
thread settings, sparsity, missing-value structure, and output requirements.

## Citation

From R:

```r
citation("bgns")
```

## License

GPL-3.
