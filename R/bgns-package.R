#' bgns: Biweight Graph and Network Statistics
#'
#' bgns provides biweight midcorrelation kernels and exact bicor-based
#' k-nearest-neighbor graph construction for dense matrices and sparse
#' \code{dgCMatrix} inputs. Computation is panelized to avoid materializing full
#' dense similarity matrices in tidy and KNN workflows. bgns builds graphs
#' from robust correlation structure, supporting workflows where similarity is
#' better represented by coordinated variation.
#'
#' @section Runtime tuning:
#' \describe{
#'   \item{\code{BGNS_MEM_MB}}{Scratch-memory budget in MB for panelization, not a total-memory limit. Default: 256.}
#'   \item{\code{BGNS_MIN_OVERLAP}}{Minimum finite overlap used by native code. Usually set through \code{min_overlap}.}
#'   \item{\code{BGNS_NUM_THREADS}}{Maximum OpenMP threads used by bgns. Default: 2 when OpenMP is available and lower if OpenMP is unavailable.}
#'   \item{\code{BGNS_BITSET_BETA}}{Controls adaptive bitset construction on sparse overlap paths. Default: 0.75.}
#'   \item{\code{BGNS_OMP_BUILD_THRESH}}{Column-count threshold before OpenMP panel building is used. Default: 8.}
#'   \item{\code{BGNS_SIMD}}{Set to 0, false, or off to disable optional SIMD kernels when compiled in.}
#' }
#'
#' @name bgns
#' @aliases bgns-package BGNS
#' @docType package
#' @useDynLib bgns, .registration = TRUE, .fixes = "C_"
#' @import Matrix
#' @import methods
#' @importFrom stats setNames
NULL
