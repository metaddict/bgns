suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_equal <- function(x, y, tolerance = 1e-8, message = "objects differ") {
  comparison <- all.equal(x, y, tolerance = tolerance, check.attributes = FALSE)
  if (!isTRUE(comparison)) {
    fail(paste(message, paste(comparison, collapse = "; "), sep = ": "))
  }
}
normalize_knn <- function(x) {
  x$col1 <- as.integer(x$col1)
  x$col2 <- as.integer(x$col2)
  x <- x[order(x$col1, x$rank, x$col2), , drop = FALSE]
  rownames(x) <- NULL
  x
}

cat("bgns:", as.character(packageVersion("bgns")), "\n")
cat("R:", R.version.string, "\n")
cat("platform:", R.version$platform, "\n")
assert_true(identical(as.character(packageVersion("bgns")), "0.4.2"),
            "installed package version is not 0.4.2")

set.seed(1201)
x <- matrix(rnorm(1200), nrow = 120, ncol = 10)
x[sample(length(x), 45)] <- NA_real_
colnames(x) <- paste0("x", seq_len(ncol(x)))
Sys.setenv(BGNS_NUM_THREADS = "1", BGNS_SIMD = "0")
cor_scalar <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
knn_scalar <- bicor_knn(x, knn = 4, pairwise.complete.obs = TRUE,
                        threshold = -Inf, min_overlap = 3)
Sys.setenv(BGNS_NUM_THREADS = "2", BGNS_SIMD = "1")
cor_default <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
knn_default <- bicor_knn(x, knn = 4, pairwise.complete.obs = TRUE,
                         threshold = -Inf, min_overlap = 3)
assert_equal(cor_scalar, cor_default, tolerance = 1e-10,
             message = "dense bicor changes with thread/SIMD settings")
assert_equal(normalize_knn(knn_scalar), normalize_knn(knn_default), tolerance = 1e-10,
             message = "dense KNN changes with thread/SIMD settings")
assert_true(identical(dim(cor_default), c(10L, 10L)),
            "dense bicor dimensions are incorrect")

tidy <- bicor(x, tidy = TRUE, threshold = 0.25, min_overlap = 3)
assert_true(all(abs(tidy$cor) >= 0.25), "tidy threshold semantics are incorrect")
for (row in seq_len(nrow(tidy))) {
  assert_equal(tidy$cor[row],
               cor_default[as.integer(tidy$col1[row]), as.integer(tidy$col2[row])],
               tolerance = 1e-8,
               message = "tidy edge differs from dense bicor")
}

complete_columns <- which(colSums(!is.finite(x)) == 0L)
knn_complete <- bicor_knn(x, knn = 3, pairwise.complete.obs = FALSE,
                          threshold = -Inf, min_overlap = 3)
assert_true(all(as.integer(knn_complete$col1) %in% complete_columns),
            "incomplete target escaped complete-observation mode")
assert_true(all(as.integer(knn_complete$col2) %in% complete_columns),
            "incomplete source escaped complete-observation mode")

set.seed(1202)
xs <- Matrix::rsparsematrix(80, 9, density = 0.35)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
cor_sparse <- bicor(xs, tidy = TRUE, threshold = -Inf, min_overlap = 1)
cor_dense_tidy <- bicor(as.matrix(xs), tidy = TRUE, threshold = -Inf, min_overlap = 1)
assert_equal(cor_sparse, cor_dense_tidy, tolerance = 1e-8,
             message = "sparse tidy bicor differs from dense reference")
knn_sparse <- bicor_knn(xs, knn = 3, threshold = -Inf, min_overlap = 1)
knn_dense <- bicor_knn(as.matrix(xs), knn = 3, threshold = -Inf, min_overlap = 1)
assert_equal(normalize_knn(knn_sparse), normalize_knn(knn_dense), tolerance = 1e-8,
             message = "sparse KNN differs from dense reference")

set.seed(1203)
y_dense <- matrix(rnorm(80 * 4), nrow = 80, ncol = 4)
colnames(y_dense) <- paste0("y", seq_len(ncol(y_dense)))
mixed_xy <- bicor_knn(xs, y_dense, knn = 3, threshold = -Inf, min_overlap = 1,
                      bipartite_levels = "separate")
reference_xy <- bicor_knn(as.matrix(xs), y_dense, knn = 3, threshold = -Inf,
                          min_overlap = 1, bipartite_levels = "separate")
assert_equal(normalize_knn(mixed_xy), normalize_knn(reference_xy), tolerance = 1e-8,
             message = "sparse-x/dense-y path differs from dense reference")
y_sparse <- Matrix::rsparsematrix(80, 4, density = 0.4)
colnames(y_sparse) <- paste0("ys", seq_len(ncol(y_sparse)))
x_dense <- matrix(rnorm(80 * 9), nrow = 80, ncol = 9)
colnames(x_dense) <- paste0("xd", seq_len(ncol(x_dense)))
mixed_yx <- bicor_knn(x_dense, y_sparse, knn = 3, threshold = -Inf,
                      min_overlap = 1, bipartite_levels = "separate")
reference_yx <- bicor_knn(x_dense, as.matrix(y_sparse), knn = 3, threshold = -Inf,
                          min_overlap = 1, bipartite_levels = "separate")
assert_equal(normalize_knn(mixed_yx), normalize_knn(reference_yx), tolerance = 1e-8,
             message = "dense-x/sparse-y path differs from dense reference")

rectangular <- bicor(x_dense, y_dense, min_overlap = 1)
assert_true(identical(dim(rectangular), c(ncol(x_dense), ncol(y_dense))),
            "rectangular bicor dimensions are incorrect")
for (i in seq_len(ncol(x_dense))) {
  for (j in seq_len(ncol(y_dense))) {
    reference <- bicor(cbind(x_dense[, i], y_dense[, j]), min_overlap = 1)[1, 2]
    assert_equal(rectangular[i, j], reference, tolerance = 1e-8,
                 message = "rectangular bicor indexing is incorrect")
  }
}

knn_stream <- bicor_knn(x_dense, knn = 3, threshold = -Inf,
                        min_overlap = 1, direct_sparse = TRUE)
knn_materialized <- bicor_knn(x_dense, knn = 3, threshold = -Inf,
                              min_overlap = 1, direct_sparse = FALSE)
assert_equal(normalize_knn(knn_stream), normalize_knn(knn_materialized), tolerance = 1e-8,
             message = "direct_sparse routes differ")

degenerate <- cbind(constant = rep(1, 20), varying = seq_len(20),
                    missing = rep(NA_real_, 20))
degenerate_cor <- bicor(degenerate, min_overlap = 3)
assert_true(identical(dim(degenerate_cor), c(3L, 3L)),
            "degenerate input returned the wrong shape")
row_error <- try(bicor_knn(matrix(1:20, 10, 2), matrix(1:18, 9, 2), knn = 1),
                 silent = TRUE)
assert_true(inherits(row_error, "try-error"), "row-count mismatch was not rejected")

cat("All focused runtime validations passed.\n")
