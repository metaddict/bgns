options(width = 120)

suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

stopifnot(identical(as.character(utils::packageVersion("bgns")), "0.4.2"))

same_numeric <- function(x, y, tolerance = 1e-9) {
  isTRUE(all.equal(x, y, tolerance = tolerance, check.attributes = TRUE))
}

normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  rownames(z) <- NULL
  z[order(z$col1, z$rank, z$col2), , drop = FALSE]
}

normalize_tidy <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  lo <- pmin(z$col1, z$col2)
  hi <- pmax(z$col1, z$col2)
  z$col1 <- lo
  z$col2 <- hi
  rownames(z) <- NULL
  z[order(z$col1, z$col2, z$cor), , drop = FALSE]
}

with_bgns_env <- function(values, code) {
  nms <- names(values)
  old <- Sys.getenv(nms, unset = NA_character_)
  on.exit({
    for (i in seq_along(nms)) {
      if (is.na(old[[i]])) Sys.unsetenv(nms[[i]]) else do.call(Sys.setenv, stats::setNames(list(old[[i]]), nms[[i]]))
    }
  }, add = TRUE)
  for (i in seq_along(nms)) {
    value <- values[[i]]
    if (is.na(value)) Sys.unsetenv(nms[[i]]) else do.call(Sys.setenv, stats::setNames(list(value), nms[[i]]))
  }
  force(code)
}

# Dense bicor with missing values, scalar/SIMD fallback, thread policy, and blocked computation.
set.seed(1001)
x_na <- matrix(rnorm(90 * 11), nrow = 90, ncol = 11)
x_na[c(4, 113, 271, 544, 799)] <- NA_real_

ref_scalar <- with_bgns_env(
  c(BGNS_NUM_THREADS = "1", BGNS_SIMD = "0", BGNS_MEM_MB = "1"),
  bicor(x_na, pairwise.complete.obs = TRUE, min_overlap = 3)
)
auto_parallel <- with_bgns_env(
  c(BGNS_NUM_THREADS = "2", BGNS_SIMD = NA_character_, BGNS_MEM_MB = "256"),
  bicor(x_na, pairwise.complete.obs = TRUE, min_overlap = 3)
)
stopifnot(same_numeric(ref_scalar, auto_parallel, tolerance = 1e-9))

# Non-pairwise semantics: incomplete columns must be ineligible.
x_incomplete <- x_na
x_incomplete[1, 2] <- NA_real_
x_incomplete[2, 7] <- Inf
complete_cols <- which(colSums(!is.finite(x_incomplete)) == 0L)
cm_complete <- bicor(x_incomplete, pairwise.complete.obs = FALSE, min_overlap = 1)
stopifnot(all(is.na(cm_complete[-complete_cols, , drop = FALSE])))
stopifnot(all(is.na(cm_complete[, -complete_cols, drop = FALSE])))

# Rectangular dense x-y indexing and pairwise numerical consistency.
set.seed(1002)
x_rect <- matrix(rnorm(60 * 7), nrow = 60, ncol = 7)
y_rect <- matrix(rnorm(60 * 4), nrow = 60, ncol = 4)
xy <- bicor(x_rect, y_rect, min_overlap = 1)
stopifnot(identical(dim(xy), c(7L, 4L)))
for (i in seq_len(ncol(x_rect))) {
  for (j in seq_len(ncol(y_rect))) {
    pair_ref <- bicor(cbind(x_rect[, i], y_rect[, j]), min_overlap = 1)[1, 2]
    stopifnot(isTRUE(all.equal(xy[i, j], pair_ref, tolerance = 1e-10)))
  }
}

# Exact dense top-k agreement with dense bicor, direct and buffered paths.
set.seed(1003)
x_dense <- matrix(rnorm(100 * 12), nrow = 100, ncol = 12)
cm <- bicor(x_dense, min_overlap = 1)
kn_direct <- bicor_knn(x_dense, knn = 4, threshold = -Inf, direct_sparse = TRUE, min_overlap = 1)
kn_buffered <- bicor_knn(x_dense, knn = 4, threshold = -Inf, direct_sparse = FALSE, min_overlap = 1)
stopifnot(same_numeric(normalize_knn(kn_direct), normalize_knn(kn_buffered), tolerance = 1e-10))
for (j in seq_len(ncol(x_dense))) {
  ord <- order(cm[, j], decreasing = TRUE, na.last = NA)
  ord <- ord[ord != j][seq_len(4)]
  got <- kn_direct[as.integer(kn_direct$col1) == j, , drop = FALSE]
  got <- got[order(got$rank), , drop = FALSE]
  stopifnot(identical(as.integer(got$col2), as.integer(ord)))
  stopifnot(isTRUE(all.equal(got$val, cm[ord, j], tolerance = 1e-9)))
}

# Determinism across thread count, SIMD setting, and memory-panel size.
kn_scalar <- with_bgns_env(
  c(BGNS_NUM_THREADS = "1", BGNS_SIMD = "0", BGNS_MEM_MB = "1"),
  bicor_knn(x_dense, knn = 4, threshold = -Inf, min_overlap = 1)
)
kn_auto <- with_bgns_env(
  c(BGNS_NUM_THREADS = "2", BGNS_SIMD = NA_character_, BGNS_MEM_MB = "256"),
  bicor_knn(x_dense, knn = 4, threshold = -Inf, min_overlap = 1)
)
stopifnot(same_numeric(normalize_knn(kn_scalar), normalize_knn(kn_auto), tolerance = 1e-9))

# Sparse tidy bicor and sparse KNN must agree with the corresponding dense matrix.
set.seed(1004)
x_sparse <- Matrix::rsparsematrix(80, 10, density = 0.35)
x_sparse_dense <- as.matrix(x_sparse)
tidy_sparse <- bicor(x_sparse, tidy = TRUE, threshold = -Inf, min_overlap = 1)
tidy_dense <- bicor(x_sparse_dense, tidy = TRUE, threshold = -Inf, min_overlap = 1)
stopifnot(same_numeric(normalize_tidy(tidy_sparse), normalize_tidy(tidy_dense), tolerance = 1e-9))

kn_sparse <- bicor_knn(x_sparse, knn = 3, threshold = -Inf, min_overlap = 1)
kn_sparse_dense <- bicor_knn(x_sparse_dense, knn = 3, threshold = -Inf, min_overlap = 1)
stopifnot(same_numeric(normalize_knn(kn_sparse), normalize_knn(kn_sparse_dense), tolerance = 1e-9))

# Coercible sparse Matrix subclasses are accepted and normalized to dgCMatrix.
x_triplet <- methods::as(x_sparse, "TsparseMatrix")
kn_triplet <- bicor_knn(x_triplet, knn = 3, threshold = -Inf, min_overlap = 1)
stopifnot(same_numeric(normalize_knn(kn_triplet), normalize_knn(kn_sparse), tolerance = 1e-9))

# Dense-sparse and sparse-dense exact bipartite parity, including separate levels.
set.seed(1005)
y_dense <- matrix(rnorm(80 * 5), nrow = 80, ncol = 5)
colnames(x_sparse) <- paste0("sx", seq_len(ncol(x_sparse)))
colnames(y_dense) <- paste0("dy", seq_len(ncol(y_dense)))

mixed_sd <- bicor_knn(x_sparse, y_dense, knn = 3, threshold = -Inf, min_overlap = 1,
                      bipartite_levels = "separate")
ref_sd <- bicor_knn(as.matrix(x_sparse), y_dense, knn = 3, threshold = -Inf, min_overlap = 1,
                    bipartite_levels = "separate")
stopifnot(same_numeric(normalize_knn(mixed_sd), normalize_knn(ref_sd), tolerance = 1e-9))

x_dense2 <- matrix(rnorm(80 * 9), nrow = 80, ncol = 9)
y_sparse <- Matrix::rsparsematrix(80, 6, density = 0.40)
colnames(x_dense2) <- paste0("dx", seq_len(ncol(x_dense2)))
colnames(y_sparse) <- paste0("sy", seq_len(ncol(y_sparse)))
mixed_ds <- bicor_knn(x_dense2, y_sparse, knn = 3, threshold = -Inf, min_overlap = 1,
                      bipartite_levels = "separate")
ref_ds <- bicor_knn(x_dense2, as.matrix(y_sparse), knn = 3, threshold = -Inf, min_overlap = 1,
                    bipartite_levels = "separate")
stopifnot(same_numeric(normalize_knn(mixed_ds), normalize_knn(ref_ds), tolerance = 1e-9))

# Sparse-sparse bipartite parity.
xs2 <- Matrix::rsparsematrix(80, 8, density = 0.45)
ys2 <- Matrix::rsparsematrix(80, 4, density = 0.50)
colnames(xs2) <- paste0("x", seq_len(ncol(xs2)))
colnames(ys2) <- paste0("y", seq_len(ncol(ys2)))
ss <- bicor_knn(xs2, ys2, knn = 2, threshold = -Inf, min_overlap = 1,
                bipartite_levels = "separate")
ss_ref <- bicor_knn(as.matrix(xs2), as.matrix(ys2), knn = 2, threshold = -Inf, min_overlap = 1,
                    bipartite_levels = "separate")
stopifnot(same_numeric(normalize_knn(ss), normalize_knn(ss_ref), tolerance = 1e-9))

# pairwise.complete.obs = FALSE must exclude incomplete source and target columns.
x_false <- x_dense
x_false[1, 3] <- NA_real_
x_false[2, 8] <- Inf
eligible <- which(colSums(!is.finite(x_false)) == 0L)
kn_false <- bicor_knn(x_false, knn = 3, pairwise.complete.obs = FALSE,
                      threshold = -Inf, min_overlap = 1)
stopifnot(all(as.integer(kn_false$col1) %in% eligible))
stopifnot(all(as.integer(kn_false$col2) %in% eligible))

# Signed threshold semantics for KNN.
kn_pos <- bicor_knn(x_dense, knn = 4, threshold = 0, min_overlap = 1)
stopifnot(all(kn_pos$val > 0))

cat(
  "Runtime validation passed:", R.version.string,
  "|", unname(Sys.info()[["sysname"]]), unname(Sys.info()[["machine"]]), "\n"
)
