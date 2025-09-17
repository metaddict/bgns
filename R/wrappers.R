# --- Package doc / imports / native registration -----------------------------

#' bgns: Biweight Graph and Network Statistics (BGNS)
#'
#' Robust biweight midcorrelation (bicor) kernels with NA-aware pairwise
#' handling, KNN/graph construction, and clustering wrappers compatible with
#' tgstat graph utilities. Optimized BLAS paths are used when inputs are NA-free
#' for speed.
#'
#' @name bgns
#' @aliases bgns-package BGNS
#' @useDynLib bgns, .registration = TRUE, .fixes = "C_"
#' @import tgstat
#' @importFrom Matrix t drop0
"_PACKAGE"

.coerce_dense <- function(x) {
  if (inherits(x, "dgCMatrix")) {
    return(as.matrix(x))
  }
  x
}

.tgs_use_blas <- function() isTRUE(getOption("tgs_use.blas", FALSE))

.bicor_finite <- function(x) {
  if (inherits(x, "dgCMatrix")) {
    return(isTRUE(all(is.finite(x@x))))
  }
  isTRUE(all(is.finite(as.matrix(x))))
}

.labels_or_index <- function(nms, n) {
  if (!is.null(nms) && length(nms) == n) nms else as.character(seq_len(n))
}

#' Robust biweight midcorrelation (bicor)
#'
#' @param x numeric matrix (rows = observations, columns = items)
#' @param y optional numeric matrix with the same number of rows as `x`
#' @param pairwise.complete.obs logical; TRUE = strict pairwise
#' @param spearman ignored (kept for tgs_cor API parity)
#' @param tidy logical; if TRUE, return tidy data.frame with (col1, col2, cor)
#' @param threshold numeric; keep pairs with |cor| >= threshold (tidy mode only)
#' @param use_intersection_denominator logical; stricter normalization
#' @return matrix (auto/cross) or tidy data.frame
#' @export
bicor <- function(
  x,
  y = NULL,
  pairwise.complete.obs = TRUE,
  spearman = FALSE, # ignored
  tidy = FALSE,
  threshold = 0,
  use_intersection_denominator = FALSE
) {
  if (missing(x)) {
    stop(
      "Usage: bicor(x, y = NULL, pairwise.complete.obs = TRUE, spearman = FALSE, tidy = FALSE, threshold = 0)",
      call. = FALSE
    )
  }
  x <- .coerce_dense(x)
  if (!is.null(y)) {
    y <- .coerce_dense(y)
  }

  if (is.null(y)) {
    C <- .Call(
      "C_bicor",
      x,
      NULL,
      isTRUE(pairwise.complete.obs),
      isTRUE(use_intersection_denominator),
      new.env(parent = parent.frame()),
      PACKAGE = "bgns"
    )
  } else {
    C <- .Call(
      "C_bicor",
      x,
      y,
      isTRUE(pairwise.complete.obs),
      isTRUE(use_intersection_denominator),
      new.env(parent = parent.frame()),
      PACKAGE = "bgns"
    )
  }

  if (!isTRUE(tidy)) {
    return(C)
  }

  # tidy formatting (tgstat style)
  if (is.null(y)) {
    p <- ncol(x)
    has_names <- !is.null(colnames(x)) && length(colnames(x)) == p
    rn <- if (has_names) colnames(x) else NULL
    keep <- which(
      upper.tri(C, diag = FALSE) & abs(C) >= threshold,
      arr.ind = TRUE
    )
    if (!nrow(keep)) {
      return(data.frame(col1 = integer(), col2 = integer(), cor = numeric()))
    }
    # Order by (row i, col j) to match tgstat's tidy ordering
    ord <- order(keep[, 1], keep[, 2])
    keep <- keep[ord, , drop = FALSE]
    i <- keep[, 1]
    j <- keep[, 2]
    if (has_names) {
      data.frame(
        col1 = factor(i, levels = seq_len(p), labels = rn),
        col2 = factor(j, levels = seq_len(p), labels = rn),
        cor = C[as.matrix(keep)],
        stringsAsFactors = TRUE
      )
    } else {
      data.frame(
        col1 = as.integer(i),
        col2 = as.integer(j),
        cor = C[as.matrix(keep)]
      )
    }
  } else {
    px <- ncol(x)
    py <- ncol(y)
    has_names_x <- !is.null(colnames(x)) && length(colnames(x)) == px
    has_names_y <- !is.null(colnames(y)) && length(colnames(y)) == py
    rn_x <- if (has_names_x) colnames(x) else NULL
    rn_y <- if (has_names_y) colnames(y) else NULL
    val <- as.double(C) # column-major over Y (groups by j)
    keep <- which(abs(val) >= threshold)
    if (!length(keep)) {
      return(data.frame(col1 = integer(), col2 = integer(), cor = numeric()))
    }
    j <- ((keep - 1L) %/% px) + 1L
    i <- ((keep - 1L) %% px) + 1L
    data.frame(
      col1 = if (has_names_x) {
        factor(i, levels = seq_len(px), labels = rn_x)
      } else {
        as.integer(i)
      },
      col2 = if (has_names_y) {
        factor(j, levels = seq_len(py), labels = rn_y)
      } else {
        as.integer(j)
      },
      cor = val[keep],
      stringsAsFactors = has_names_x || has_names_y
    )
  }
}

#' KNN by bicor
#'
#' @param x numeric matrix (rows = observations, columns = items)
#' @param y optional numeric matrix with the same number of rows as `x`
#' @param knn neighbors per target
#' @param pairwise.complete.obs logical; TRUE = strict pairwise
#' @param threshold numeric; keep edges >= threshold before K selection
#' @param use_intersection_denominator logical
#' @return data.frame with columns col1, col2, val, rank
#' @export
bicor_knn <- function(
  x,
  y = NULL,
  knn,
  pairwise.complete.obs = TRUE,
  threshold = 0,
  use_intersection_denominator = FALSE
) {
  if (missing(x) || missing(knn)) {
    stop(
      "Usage: bicor_knn(x, y, knn, pairwise.complete.obs = TRUE, threshold = 0)",
      call. = FALSE
    )
  }
  x <- .coerce_dense(x)
  if (!is.null(y)) {
    y <- .coerce_dense(y)
  }

  if (is.null(y)) {
    .Call(
      "C_bicor_knn",
      x,
      NULL,
      as.integer(knn),
      as.numeric(threshold),
      isTRUE(use_intersection_denominator),
      new.env(parent = parent.frame()),
      PACKAGE = "bgns"
    )
  } else {
    .Call(
      "C_bicor_knn",
      x,
      y,
      as.integer(knn),
      as.numeric(threshold),
      isTRUE(use_intersection_denominator),
      new.env(parent = parent.frame()),
      PACKAGE = "bgns"
    )
  }
}

# #' Bicor-based graph construction
# #' @param x numeric matrix (rows = observations, columns = items)
# #' @param knn K nearest neighbors
# #' @param k_expand expansion factor used by tgstat's graph builder
# #' @param k_beta numeric (default 3); passed to tgstat if applicable
# #' @param use_intersection_denominator logical; passed to bicor()
# #' @return graph object created by tgstat
# #' @export
# bicor_graph <- function(
#   x,
#   knn,
#   k_expand,
#   k_beta = 3,
#   use_intersection_denominator = FALSE
# ) {
#   if (missing(x) || missing(knn) || missing(k_expand)) {
#     stop("Usage: bicor_graph(x, knn, k_expand, k_beta = 3)", call. = FALSE)
#   }
#   x <- .coerce_dense(x)
#   S <- bicor(
#     x,
#     pairwise.complete.obs = TRUE,
#     use_intersection_denominator = use_intersection_denominator
#   )
#   .Call(
#     "tgs_cor_graph",
#     S,
#     as.integer(knn),
#     as.integer(k_expand),
#     as.numeric(k_beta),
#     new.env(parent = parent.frame()),
#     PACKAGE = "tgstat"
#   )
# }

# # #' Bicor-based graph clustering
# #' @param graph graph returned by bicor_graph()/tgs_graph()
# #' @param min_cluster_size integer
# #' @param cooling numeric; default 1.05
# #' @param burn_in integer; default 10
# #' @return clustering object returned by tgstat
# #' @export
# bicor_graph_cover <- function(
#   graph,
#   min_cluster_size,
#   cooling = 1.05,
#   burn_in = 10
# ) {
#   if (missing(graph) || missing(min_cluster_size)) {
#     stop(
#       "Usage: bicor_graph_cover(graph, min_cluster_size, cooling = 1.05, burn_in = 10)",
#       call. = FALSE
#     )
#   }
#   .Call(
#     "tgs_graph2cluster",
#     graph,
#     as.integer(min_cluster_size),
#     as.numeric(cooling),
#     as.integer(burn_in),
#     new.env(parent = parent.frame()),
#     PACKAGE = "tgstat"
#   )
# }

# #' Bicor-based graph clustering with resampling (ensemble)
# #' @param graph graph returned by bicor_graph()/tgs_graph()
# #' @param knn K used in the graph
# #' @param min_cluster_size integer
# #' @param cooling numeric; default 1.05
# #' @param burn_in integer; default 10
# #' @param p_resamp resampling proportion (0,1]
# #' @param n_resamp number of resamples
# #' @param method one of "hash","full","edges"
# #' @return resampled clustering object returned by tgstat
# #' @export
# bicor_graph_cover_resample <- function(
#   graph,
#   knn,
#   min_cluster_size,
#   cooling = 1.05,
#   burn_in = 10,
#   p_resamp = 0.75,
#   n_resamp = 500,
#   method = c("hash", "full", "edges")
# ) {
#   if (missing(graph) || missing(knn) || missing(min_cluster_size)) {
#     stop(
#       "Usage: bicor_graph_cover_resample(graph, knn, min_cluster_size, cooling = 1.05, burn_in = 10, p_resamp = 0.75, n_resamp = 500)",
#       call. = FALSE
#     )
#   }
#   method <- match.arg(method)

#   if (method == "hash") {
#     .Call(
#       "tgs_graph2cluster_multi_hash",
#       graph,
#       as.integer(knn),
#       as.integer(min_cluster_size),
#       as.numeric(cooling),
#       as.integer(burn_in),
#       as.numeric(p_resamp),
#       as.integer(n_resamp),
#       new.env(parent = parent.frame()),
#       PACKAGE = "tgstat"
#     )
#   } else if (method == "full") {
#     .Call(
#       "tgs_graph2cluster_multi_full",
#       graph,
#       as.integer(knn),
#       as.integer(min_cluster_size),
#       as.numeric(cooling),
#       as.integer(burn_in),
#       as.numeric(p_resamp),
#       as.integer(n_resamp),
#       new.env(parent = parent.frame()),
#       PACKAGE = "tgstat"
#     )
#   } else {
#     # "edges"
#     .Call(
#       "tgs_graph2cluster_multi_edges",
#       graph,
#       as.integer(knn),
#       as.integer(min_cluster_size),
#       as.numeric(cooling),
#       as.integer(burn_in),
#       as.numeric(p_resamp),
#       as.integer(n_resamp),
#       new.env(parent = parent.frame()),
#       PACKAGE = "tgstat"
#     )
#   }
# }
