suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

stopifnot(as.character(utils::packageVersion("bgns")) == "0.4.2")

normalize_tidy <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z <- z[order(z$col1, z$col2, z$cor), , drop = FALSE]
  rownames(z) <- NULL
  z
}

normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z <- z[order(z$col1, z$rank, z$col2), , drop = FALSE]
  rownames(z) <- NULL
  z
}

expect_equal_numeric <- function(x, y, tolerance = 1e-8, label = "objects") {
  comparison <- all.equal(x, y, tolerance = tolerance, check.attributes = TRUE)
  if (!isTRUE(comparison)) {
    stop(label, " are not equal within tolerance ", tolerance, ": ",
         paste(comparison, collapse = "; "), call. = FALSE)
  }
}

old_env <- Sys.getenv(c("BGNS_NUM_THREADS", "BGNS_MEM_MB"), unset = NA_character_)
on.exit({
  for (nm in names(old_env)) {
    if (is.na(old_env[[nm]])) {
      Sys.unsetenv(nm)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_env[[nm]]), nm))
    }
  }
}, add = TRUE)

# Dense execution must be stable across one and two OpenMP threads. SIMD is
# controlled at process startup by the workflow, because ISA detection is cached.
set.seed(1001)
x <- matrix(rnorm(2400), nrow = 120, ncol = 20)
x[sample(length(x), 60)] <- NA_real_

Sys.setenv(BGNS_NUM_THREADS = "1", BGNS_MEM_MB = "1")
dense_serial <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
knn_serial <- bicor_knn(x, knn = 5, pairwise.complete.obs = TRUE,
                        threshold = -Inf, min_overlap = 3,
                        direct_sparse = TRUE)

Sys.setenv(BGNS_NUM_THREADS = "2", BGNS_MEM_MB = "8")
dense_parallel <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
knn_parallel <- bicor_knn(x, knn = 5, pairwise.complete.obs = TRUE,
                          threshold = -Inf, min_overlap = 3,
                          direct_sparse = TRUE)

expect_equal_numeric(dense_serial, dense_parallel, 1e-10,
                     "dense one-thread and two-thread results")
expect_equal_numeric(normalize_knn(knn_serial), normalize_knn(knn_parallel),
                     1e-10, "KNN one-thread and two-thread results")

# Streaming and materialized KNN implementations must agree.
knn_materialized <- bicor_knn(x, knn = 5, pairwise.complete.obs = TRUE,
                              threshold = -Inf, min_overlap = 3,
                              direct_sparse = FALSE)
expect_equal_numeric(normalize_knn(knn_parallel), normalize_knn(knn_materialized),
                     1e-10, "streaming and materialized KNN results")

# Rectangular dense x-y correlation must have correct dimensions, names, values.
set.seed(1002)
xd <- matrix(rnorm(420), nrow = 60, ncol = 7,
             dimnames = list(NULL, paste0("x", 1:7)))
yd <- matrix(rnorm(240), nrow = 60, ncol = 4,
             dimnames = list(NULL, paste0("y", 1:4)))
xy <- bicor(xd, yd, min_overlap = 1)
stopifnot(identical(dim(xy), c(7L, 4L)))
stopifnot(identical(rownames(xy), colnames(xd)))
stopifnot(identical(colnames(xy), colnames(yd)))
for (i in seq_len(ncol(xd))) {
  for (j in seq_len(ncol(yd))) {
    ref <- bicor(cbind(xd[, i], yd[, j]), min_overlap = 1)[1, 2]
    if (!isTRUE(all.equal(xy[i, j], ref, tolerance = 1e-10))) {
      stop("rectangular dense bicor indexing/value mismatch", call. = FALSE)
    }
  }
}

# Tidy dense output must reproduce the corresponding dense matrix entries.
dense_tidy <- bicor(xd, tidy = TRUE, threshold = -Inf, min_overlap = 1)
dense_full <- bicor(xd, min_overlap = 1)
for (row in seq_len(nrow(dense_tidy))) {
  i <- as.integer(dense_tidy$col1[row])
  j <- as.integer(dense_tidy$col2[row])
  if (!isTRUE(all.equal(dense_tidy$cor[row], dense_full[i, j],
                        tolerance = 1e-10))) {
    stop("tidy dense output does not match dense bicor", call. = FALSE)
  }
}

# Sparse tidy bicor must agree with the dense representation of the same matrix.
set.seed(1003)
xs <- Matrix::rsparsematrix(90, 14, density = 0.25)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
sparse_tidy <- bicor(xs, tidy = TRUE, threshold = -Inf, min_overlap = 1)
dense_sparse_ref <- bicor(as.matrix(xs), tidy = TRUE,
                          threshold = -Inf, min_overlap = 1)
expect_equal_numeric(normalize_tidy(sparse_tidy),
                     normalize_tidy(dense_sparse_ref), 1e-8,
                     "sparse and dense tidy bicor results")

# Sparse classes coercible to dgCMatrix must be accepted.
xt <- suppressWarnings(methods::as(xs, "TsparseMatrix"))
triplet_tidy <- suppressWarnings(
  bicor(xt, tidy = TRUE, threshold = -Inf, min_overlap = 1)
)
expect_equal_numeric(normalize_tidy(triplet_tidy), normalize_tidy(sparse_tidy),
                     1e-8, "triplet-coerced and dgCMatrix results")

# Sparse-sparse and mixed dense/sparse KNN must agree with dense references.
set.seed(1004)
ys <- Matrix::rsparsematrix(90, 6, density = 0.30)
colnames(ys) <- paste0("t", seq_len(ncol(ys)))
ss <- bicor_knn(xs, ys, knn = 3, threshold = -Inf, min_overlap = 1,
                bipartite_levels = "separate")
ss_ref <- bicor_knn(as.matrix(xs), as.matrix(ys), knn = 3,
                    threshold = -Inf, min_overlap = 1,
                    bipartite_levels = "separate")
expect_equal_numeric(normalize_knn(ss), normalize_knn(ss_ref), 1e-8,
                     "sparse-sparse and dense KNN results")

mixed_sd <- bicor_knn(xs, as.matrix(ys), knn = 3, threshold = -Inf,
                      min_overlap = 1, bipartite_levels = "separate")
mixed_ds <- bicor_knn(as.matrix(xs), ys, knn = 3, threshold = -Inf,
                      min_overlap = 1, bipartite_levels = "separate")
expect_equal_numeric(normalize_knn(mixed_sd), normalize_knn(ss_ref), 1e-8,
                     "sparse-dense and dense KNN results")
expect_equal_numeric(normalize_knn(mixed_ds), normalize_knn(ss_ref), 1e-8,
                     "dense-sparse and dense KNN results")

# pairwise.complete.obs = FALSE must exclude incomplete source/target columns.
set.seed(1005)
xmiss <- matrix(rnorm(600), nrow = 60, ncol = 10)
xmiss[1, 2] <- NA_real_
xmiss[2, 7] <- Inf
complete <- which(colSums(!is.finite(xmiss)) == 0L)
kn_complete <- bicor_knn(xmiss, knn = 4, pairwise.complete.obs = FALSE,
                         threshold = -Inf, min_overlap = 1)
stopifnot(all(as.integer(kn_complete$col1) %in% complete))
stopifnot(all(as.integer(kn_complete$col2) %in% complete))

# Integer matrices, constant columns, and low-overlap columns must not crash.
xi <- matrix(sample.int(20, 600, replace = TRUE), nrow = 60, ncol = 10)
xi[, 10] <- 4L
xi[1:58, 9] <- NA_integer_
ci <- bicor(xi, pairwise.complete.obs = TRUE, min_overlap = 3)
stopifnot(identical(dim(ci), c(10L, 10L)))
stopifnot(is.na(ci[10, 1]) || is.finite(ci[10, 1]))
ki <- bicor_knn(xi, knn = 3, threshold = -Inf, min_overlap = 3)
stopifnot(is.data.frame(ki))

# Signed KNN and absolute tidy threshold semantics.
positive_edges <- bicor_knn(xd, knn = 4, threshold = 0, min_overlap = 1)
stopifnot(all(positive_edges$val > 0))
tidy_abs <- bicor(xd, tidy = TRUE, threshold = 0.15, min_overlap = 1)
stopifnot(all(abs(tidy_abs$cor) >= 0.15))

# Intersection-denominator paths must return finite-or-NA correlations.
inter_dense <- bicor(x, tidy = TRUE, threshold = -Inf,
                     use_intersection_denominator = TRUE, min_overlap = 3)
inter_sparse <- bicor(xs, tidy = TRUE, threshold = -Inf,
                      use_intersection_denominator = TRUE, min_overlap = 1)
stopifnot(all(is.finite(inter_dense$cor) | is.na(inter_dense$cor)))
stopifnot(all(is.finite(inter_sparse$cor) | is.na(inter_sparse$cor)))

# Row-count mismatch must be rejected cleanly.
row_error <- try(bicor_knn(matrix(1:20, 10, 2), matrix(1:18, 9, 2), knn = 1),
                 silent = TRUE)
stopifnot(inherits(row_error, "try-error"))

signature <- list(
  dense = dense_parallel,
  knn = normalize_knn(knn_parallel),
  rectangular = xy,
  sparse_tidy = normalize_tidy(sparse_tidy),
  sparse_knn = normalize_knn(ss),
  mixed_sparse_dense = normalize_knn(mixed_sd),
  mixed_dense_sparse = normalize_knn(mixed_ds)
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1L && nzchar(args[[1L]])) {
  saveRDS(signature, args[[1L]], version = 3)
}

cat("Runtime validation passed on ", R.version.string, " / ",
    Sys.info()[["sysname"]], "\n", sep = "")
