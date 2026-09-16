.is_dgc <- function(x) inherits(x, "dgCMatrix")
.is_sparse_matrix <- function(x) inherits(x, "sparseMatrix")
.as_dgc_if_sparse <- function(x, name) {
  if (.is_dgc(x) || !.is_sparse_matrix(x)) return(x)

  # Matrix >= 1.7 deprecates direct coercion from some concrete sparse
  # classes (for example dgTMatrix) straight to dgCMatrix. Follow Matrix's
  # virtual-class coercion path instead. The native backend accepts numeric
  # double sparse matrices only, matching the package's documented input type.
  if (!inherits(x, "dMatrix")) {
    stop(sprintf("%s must be a numeric sparse matrix", name), call. = FALSE)
  }
  out <- methods::as(x, "CsparseMatrix")
  if (!inherits(out, "generalMatrix")) {
    out <- methods::as(out, "generalMatrix")
  }
  if (!inherits(out, "dgCMatrix")) {
    stop(sprintf("%s could not be converted to a numeric dgCMatrix", name),
         call. = FALSE)
  }
  out
}

.bgns_scalar_logical <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("%s must be TRUE or FALSE", name), call. = FALSE)
  }
  x
}

.bgns_scalar_number <- function(x, name, allow_infinite = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) ||
      (!allow_infinite && !is.finite(x))) {
    stop(sprintf("%s must be a single %s number", name,
                 if (allow_infinite) "non-missing" else "finite"),
         call. = FALSE)
  }
  x
}

.bgns_positive_integer <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x < 1 || x != floor(x) || x > .Machine$integer.max) {
    stop(sprintf("%s must be a single positive integer", name), call. = FALSE)
  }
  as.integer(x)
}

.bgns_with_envvar <- function(name, value, thunk) {
  old <- Sys.getenv(name, unset = NA_character_)
  had <- !is.na(old)
  on.exit({
    if (had) do.call(Sys.setenv, stats::setNames(list(old), name)) else Sys.unsetenv(name)
  }, add = TRUE)
  if (is.null(value)) Sys.unsetenv(name) else do.call(Sys.setenv, stats::setNames(list(value), name))
  thunk()
}

#' Robust biweight midcorrelation
#'
#' @param x numeric matrix with rows as observations and columns as items, or a
#'   sparse `dgCMatrix` when `tidy = TRUE`.
#' @param y optional numeric matrix or `dgCMatrix` with the same number of rows
#'   as `x`.
#' @param pairwise.complete.obs logical. If `TRUE`, correlations are computed on
#'   finite pairwise overlap. If `FALSE`, pairs involving incomplete columns are
#'   omitted or returned as `NA`, depending on output mode.
#' @param spearman logical; accepted for API compatibility and ignored.
#' @param tidy logical. If `TRUE`, return a tidy edge table with columns `col1`,
#'   `col2`, and `cor`.
#' @param threshold numeric. In tidy mode, keep pairs with `abs(cor) >= threshold`.
#' @param use_intersection_denominator logical. For sparse or incomplete columns,
#'   use the overlap-specific denominator rather than full-column sums of squares.
#' @param min_overlap integer. Minimum number of shared finite observations
#'   required between two columns, including fully finite paths.
#' @return A dense correlation matrix for dense inputs when `tidy = FALSE`, or a
#'   data frame with columns `col1`, `col2`, and `cor` when `tidy = TRUE`.
#' @export
bicor <- function(
  x,
  y = NULL,
  pairwise.complete.obs = TRUE,
  spearman = FALSE,
  tidy = FALSE,
  threshold = -Inf,
  use_intersection_denominator = FALSE,
  min_overlap = 3
) {
  if (missing(x)) stop("bicor: 'x' is required", call. = FALSE)
  x <- .as_dgc_if_sparse(x, "x")
  if (!is.null(y)) y <- .as_dgc_if_sparse(y, "y")

  pairwise.complete.obs <- .bgns_scalar_logical(pairwise.complete.obs, "pairwise.complete.obs")
  tidy <- .bgns_scalar_logical(tidy, "tidy")
  use_intersection_denominator <- .bgns_scalar_logical(
    use_intersection_denominator,
    "use_intersection_denominator"
  )
  threshold <- .bgns_scalar_number(threshold, "threshold", allow_infinite = TRUE)
  min_overlap <- .bgns_positive_integer(min_overlap, "min_overlap")

  if (!identical(spearman, FALSE)) {
    warning("spearman is accepted for API compatibility but ignored; bicor computes biweight midcorrelation.", call. = FALSE)
  }

  if (isTRUE(tidy)) {
    return(.bgns_with_envvar("BGNS_MIN_OVERLAP", min_overlap, function() .Call(
      C_C_bicor_tidy,
      x,
      if (is.null(y)) NULL else y,
      pairwise.complete.obs,
      threshold,
      use_intersection_denominator,
      new.env(parent = parent.frame())
    )))
  }

  if (.is_dgc(x) || (!is.null(y) && .is_dgc(y))) {
    stop("bicor(tidy = FALSE) requires dense matrices. Use tidy = TRUE for sparse inputs or explicitly coerce.", call. = FALSE)
  }

  .bgns_with_envvar("BGNS_MIN_OVERLAP", min_overlap, function() .Call(
    C_C_bicor,
    x,
    if (is.null(y)) NULL else y,
    pairwise.complete.obs,
    use_intersection_denominator,
    new.env(parent = parent.frame())
  ))
}

#' k-nearest neighbors by biweight midcorrelation
#'
#' @param x numeric matrix or `dgCMatrix` with rows as observations and columns
#'   as source items.
#' @param y optional numeric matrix or `dgCMatrix` with the same number of rows
#'   as `x`. When supplied, neighbors are selected from columns of `x` for each
#'   target column of `y`.
#' @param knn integer. Number of neighbors per target column.
#' @param pairwise.complete.obs logical. If `TRUE`, compute each candidate edge on
#'   the finite overlap for that column pair. If `FALSE`, only pairs where both
#'   preprocessed columns are complete are eligible.
#' @param threshold numeric. Signed cutoff applied before top-k selection;
#'   candidate bicor values must be strictly greater than `threshold`.
#' @param use_intersection_denominator logical. Use overlap-specific denominators
#'   for incomplete or sparse paths.
#' @param direct_sparse logical. If `TRUE`, stream top-k edges without dense
#'   similarity intermediates.
#' @param bipartite_levels one of `"strict"` or `"separate"`. Use `"separate"`
#'   when `x` and `y` have distinct column-name levels.
#' @param min_overlap integer. Minimum number of shared finite observations
#'   required between two columns, including fully finite paths.
#' @return A data frame with columns `col1`, `col2`, `val`, and `rank`.
#' @export
bicor_knn <- function(
  x,
  y = NULL,
  knn,
  pairwise.complete.obs = TRUE,
  threshold = -Inf,
  use_intersection_denominator = FALSE,
  direct_sparse = TRUE,
  bipartite_levels = c("strict", "separate"),
  min_overlap = 3
) {
  if (missing(x) || missing(knn)) stop("bicor_knn: 'x' and 'knn' are required", call. = FALSE)
  x <- .as_dgc_if_sparse(x, "x")
  if (!is.null(y)) y <- .as_dgc_if_sparse(y, "y")

  knn <- .bgns_positive_integer(knn, "knn")
  pairwise.complete.obs <- .bgns_scalar_logical(pairwise.complete.obs, "pairwise.complete.obs")
  threshold <- .bgns_scalar_number(threshold, "threshold", allow_infinite = TRUE)
  use_intersection_denominator <- .bgns_scalar_logical(
    use_intersection_denominator,
    "use_intersection_denominator"
  )
  direct_sparse <- .bgns_scalar_logical(direct_sparse, "direct_sparse")
  min_overlap <- .bgns_positive_integer(min_overlap, "min_overlap")
  bipartite_levels <- match.arg(bipartite_levels)

  thunk_dense <- function() {
    .Call(
      C_C_bicor_knn_opts,
      x,
      if (is.null(y)) NULL else y,
      knn,
      pairwise.complete.obs,
      threshold,
      use_intersection_denominator,
      direct_sparse,
      new.env(parent = parent.frame())
    )
  }
  thunk_csc <- function() {
    .Call(
      C_C_bicor_knn_csc,
      x,
      if (is.null(y)) NULL else y,
      knn,
      pairwise.complete.obs,
      threshold,
      use_intersection_denominator,
      direct_sparse,
      new.env(parent = parent.frame())
    )
  }
  thunk_spdem <- function() {
    .Call(
      C_C_bicor_knn_spdem,
      x,
      y,
      knn,
      pairwise.complete.obs,
      threshold,
      use_intersection_denominator,
      direct_sparse,
      new.env(parent = parent.frame())
    )
  }

  wrap_minov <- function(thunk) .bgns_with_envvar("BGNS_MIN_OVERLAP", min_overlap, thunk)
  wrap_levels <- function(thunk) {
    if (!is.null(y) && identical(bipartite_levels, "separate")) {
      .bgns_with_envvar("BGNS_KNN_BIPARTITE_LEVELS", "separate", function() wrap_minov(thunk))
    } else {
      wrap_minov(thunk)
    }
  }

  x_dgc <- .is_dgc(x)
  y_dgc <- !is.null(y) && .is_dgc(y)

  if (is.null(y)) {
    if (x_dgc) wrap_levels(thunk_csc) else wrap_levels(thunk_dense)
  } else if (x_dgc && y_dgc) {
    wrap_levels(thunk_csc)
  } else if (!x_dgc && !y_dgc) {
    wrap_levels(thunk_dense)
  } else {
    wrap_levels(thunk_spdem)
  }
}
