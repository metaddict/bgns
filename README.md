# bgns: Biweight Graph and Network Statistics

`bgns` provides biweight midcorrelation (`bicor`) and exact bicor-based
k-nearest-neighbor graph construction for high-dimensional numeric matrices.
Its C++17 backend supports dense, sparse, and mixed-input computation, with
panelized similarity evaluation and bounded top-k retention for KNN output.

Biweight midcorrelation uses robust location and scale estimates to reduce
the influence of outlying observations. The package was developed for
correlation-based analyses of gene expression, single-cell and metacell data,
and other biological measurements, but its input is not restricted to
biological matrices.

`bgns` provides two core functions: `bicor()` computes correlation matrices
or edge tables, and `bicor_knn()` constructs directed KNN graphs. These
outputs can be used for downstream clustering, visualization, and biological
annotation with other packages.

## Computational design

The implementation separates similarity evaluation from retention of the
results. Dense kernels use BLAS-backed computation where applicable; sparse
and mixed-input paths handle compressed sparse matrices and finite-observation
overlap. Panelization limits the size of intermediate calculations, while
adaptive overlap strategies include index intersections and bitsets. SIMD
kernels and OpenMP parallelism are used where supported by the build and
execution environment.

With the default `direct_sparse = TRUE`, `bicor_knn()` evaluates candidate
similarities in panels and retains bounded top-k candidate sets rather than
materializing the full dense similarity matrix. For self-KNN on `p` items,
the returned graph contains at most `p * knn` edges, excluding self-neighbors.
This reduces similarity-storage requirements; it does not make the search
approximate or remove the quadratic number of candidate pairs in self-KNN.

Dense correlation output still requires storage for the full result matrix.
A correlation edge table can also grow quadratically when its threshold
retains most pairs. The panel-memory setting guides temporary allocations;
it is not a cap on total process memory, which also includes input, output,
preprocessing, and library workspace.

## Installation

Requires R 4.3.0 or later.

```r
install.packages("remotes")
remotes::install_github("metaddict/bgns")
```

GitHub installation builds the package from source. A C++17-capable toolchain
and the libraries required by your R installation are therefore necessary.
Use the matching [Rtools](https://cran.r-project.org/bin/windows/Rtools/) on
Windows or the recommended [R for macOS build tools](https://mac.r-project.org/tools/)
on macOS. OpenMP is optional; a build without it uses serial execution for
BGNS's own parallel regions. WGCNA is not required for normal use.

## Usage

Rows are observations and columns are the items being compared. For a
cell-cell graph, cells should be columns; for a gene-gene graph, genes should
be columns. Input normalization and feature selection are determined by the
analysis, not performed automatically by `bgns`.

```r
library(bgns)

set.seed(1)
x <- matrix(rnorm(100 * 8), nrow = 100, ncol = 8)
colnames(x) <- paste0("item_", seq_len(ncol(x)))

# Introduce a correlated pair and one missing observation.
x[, 2] <- 0.8 * x[, 1] + 0.2 * rnorm(nrow(x))
x[1, 3] <- NA_real_
```

### Correlation matrix

```r
cm <- bicor(x, min_overlap = 3)
round(cm, 3)
```

The result contains correlations between columns. With dense `x` and `y`,
`bicor(x, y)` returns a matrix with `ncol(x)` rows and `ncol(y)` columns.
Dense matrix output requires dense inputs; use `tidy = TRUE` for sparse
inputs.

### Correlation edges

```r
edges <- bicor(x, tidy = TRUE, threshold = 0.2)
head(edges)
```

The returned data frame has columns `col1`, `col2`, and `cor`. In this mode,
`threshold` is applied to the absolute score: the example retains finite
correlations with `abs(cor) >= 0.2`, including sufficiently strong negative
correlations.

### Exact KNN graph

```r
kn <- bicor_knn(x, knn = 3, threshold = 0)
head(kn)
```

The returned data frame has columns `col1`, `col2`, `val`, and `rank`.
`col1` identifies the target, `col2` a selected neighbor, `val` the bicor
score, and `rank` the neighbor's position within that target's results.

KNN selection ranks signed bicor scores in descending order. Unlike the
absolute cutoff in `bicor(tidy = TRUE)`, the KNN cutoff is strictly
`val > threshold`. Thus, `threshold = 0` retains only positive correlations.
The default `threshold = -Inf` permits all finite scores, including negative
ones.

Each target has at most `knn` neighbors. Fewer are returned when insufficient
eligible candidates pass the cutoff. Self-neighbors are excluded in
self-KNN, and reciprocal edges are not enforced.

## Sparse and mixed inputs

Sparse inputs are handled through numeric double compressed-column matrices
of class `dgCMatrix`. Other supported numeric sparse `Matrix` classes are
converted to this representation. Users do not need to densify the entire
input, although internal computation may use dense panels.

```r
# Construct a sparse representation of the example data.
set.seed(2)
xs <- x
xs[sample.int(length(xs), size = floor(0.2 * length(xs)))] <- 0
xs <- Matrix::Matrix(xs, sparse = TRUE)

edges_sparse <- bicor(xs, tidy = TRUE, threshold = 0.2)
kn_sparse <- bicor_knn(xs, knn = 3, threshold = 0)
```

Implicit sparse entries are numerical zeros, not missing observations.
Sparse storage does not by itself guarantee that a column has a defined
biweight scale; see the missing-data and degenerate-column conventions below.

### Bipartite KNN

With `y` supplied, neighbors are selected from columns of `x` for each target
column of `y`. The inputs must have the same number of rows, representing the
same observations in the same order. Either input may be dense or sparse.

```r
set.seed(3)
y <- matrix(rnorm(nrow(xs) * 4), nrow = nrow(xs), ncol = 4)
colnames(y) <- paste0("target_", seq_len(ncol(y)))

kn_xy <- bicor_knn(
  x = xs,
  y = y,
  knn = 3,
  bipartite_levels = "separate"
)

head(kn_xy)
```

Here, `col1` refers to targets in `y`, and `col2` refers to selected neighbors
in `x`. Use `bipartite_levels = "separate"` when the two inputs have distinct
column-name sets.

## Missing data and degenerate columns

With `pairwise.complete.obs = TRUE`, candidate pairs are evaluated on their
shared finite observations. `NA`, `NaN`, and infinite values are non-finite;
zeros remain observations. `min_overlap` defaults to 3 and also applies to
fully finite pairs.

Robust location and scale are estimated separately for each column from its
finite observations, not re-estimated on each pair's overlap. By default,
`use_intersection_denominator = FALSE` uses full-column sums of squared
biweight-transformed values for normalization. Setting it to `TRUE`
restricts the denominator to the shared finite observations as well. This
changes normalization, not the columnwise location and scale estimates.

With `pairwise.complete.obs = FALSE`, columns containing non-finite values
are ineligible. Affected correlations are `NA` in dense output and omitted
from edge and KNN output.

Constant columns, all-missing columns, and columns with zero median absolute
deviation or otherwise undefined biweight normalization produce undefined
correlations. A nonconstant column can also have zero median absolute
deviation, particularly when many observations are identical, as in
zero-heavy data. Undefined pairs are omitted from edge and KNN results. In
dense self-correlation output, ineligible diagonal entries remain `NA`.

## Runtime configuration

Set environment controls before the first computation in an R session.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BGNS_MEM_MB` | `256` | Guides the scratch-memory budget used for panel sizing; not a total-memory limit. |
| `BGNS_NUM_THREADS` | At most `2` | Limits BGNS OpenMP threads when OpenMP is available. |
| `BGNS_SIMD` | Build-dependent | Set to `0`, `false`, or `off` to disable the package's compiled SIMD paths. |
| `BGNS_BITSET_BETA` | `0.75` | Controls the adaptive criterion for constructing finite-overlap bitsets. |

```r
Sys.setenv(BGNS_MEM_MB = "256", BGNS_NUM_THREADS = "2")
```

SIMD and OpenMP availability depend on the compiler target and installed
build. The BGNS thread setting does not control independent BLAS threading.
Use the exported `min_overlap` argument rather than setting
`BGNS_MIN_OVERLAP` directly.

## Documentation

```r
help("bicor", package = "bgns")
help("bicor_knn", package = "bgns")
```

The source vignette is available in
[`vignettes/bgns.Rmd`](vignettes/bgns.Rmd).

## Citation

```r
citation("bgns")
```

## License

`bgns` is distributed under the
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.html).