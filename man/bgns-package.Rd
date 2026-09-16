\name{bgns-package}
\alias{bgns-package}
\alias{bgns}
\alias{BGNS}
\title{Biweight Graph and Network Statistics}
\description{
\pkg{bgns} provides biweight midcorrelation kernels and exact bicor-based
k-nearest-neighbor graph construction for dense matrices and sparse
\code{dgCMatrix} inputs.
}
\details{
bgns builds graphs from robust correlation structure, supporting workflows
where similarity is better represented by coordinated variation. Tidy and KNN
workflows are panelized so that a full dense similarity matrix does not need to
be materialized.

Rows are observations and columns are items. In a KNN result, \code{col1} is the
target column, \code{col2} is the selected neighbor, \code{val} is the bicor
score, and \code{rank} is the within-target neighbor rank.

Runtime can be tuned with environment variables. \env{BGNS_MEM_MB} controls the
scratch-memory budget in MB for panelization (default 256); it is not a limit
on total process memory, which also includes inputs, outputs and other allocations. \env{BGNS_NUM_THREADS} controls
OpenMP threads when available and defaults to at most two threads. \env{BGNS_SIMD} can be set to 0, false, or off to disable optional SIMD kernels when they are compiled in.
\env{BGNS_BITSET_BETA} controls sparse overlap bitset construction (default 0.75).
\env{BGNS_OMP_BUILD_THRESH} is the column-count threshold for parallel panel
building (default 8). Set \code{min_overlap} through the function argument;
it temporarily controls the native \env{BGNS_MIN_OVERLAP} setting.
OpenMP settings do not control threads used independently by the BLAS library.
}
\seealso{\code{\link{bicor}}, \code{\link{bicor_knn}}}
\author{Aditya Kshirsagar}
