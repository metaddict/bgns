#' bgns: Biweight Graph and Network Statistics (BGNS)
#'
#' Robust biweight midcorrelation (bicor) kernels with NA-aware pairwise
#' handling, KNN/graph construction, and clustering wrappers compatible
#' with tgstat graph utilities. Optimized BLAS paths are used when inputs
#' are NA-free for speed.
#'
#' @name bgns
#' @aliases bgns-package BGNS
#' @useDynLib bgns, .registration = TRUE, .fixes = "C_"
#' @import tgstat
#' @importFrom Matrix t drop0
"_PACKAGE"