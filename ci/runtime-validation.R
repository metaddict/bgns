options(warn = 2)
suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

fail <- function(...) stop(sprintf(...), call. = FALSE)
assert_true <- function(x, msg) {
  if (!isTRUE(x)) fail("%s", msg)
}
assert_equal <- function(x, y, tolerance = 1e-8, msg = "objects differ") {
  out <- all.equal(x, y, tolerance = tolerance, check.attributes = FALSE)
  if (!isTRUE(out)) fail("%s: %s", msg, paste(out, collapse = "; "))
}
normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  rownames(z) <- NULL
  z[order(z$col1, z$rank, z$col2), , drop = FALSE]
}

cat("bgns version:", as.character(packageVersion("bgns")), "\n")
cat("R version:", R.version.string, "\n")
cat("platform:", R.version$platform, "\n")
assert_true(identical(as.character(packageVersion("bgns")), "0.4.2"), "installed version is not 0.4.2")

set.seed(1201)
x <- matrix(rnorm(1200), nrow = 120, ncol = 10)
x[sample(length(x), 45)] <- NA_real_
colnames(x) <- paste0("x", seq_len(ncol(x)))

# Dense path and deterministic thread behavior.
Sys.setenv(BGNS_NUM_THREADS = "1", BGNS_SIMD = "0")
c1 <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
k1 <- bicor_knn(x, knn = 4, pairwise.complete.obs = TRUE,
                threshold = -Inf, min_overlap = 3)
Sys.setenv(BGNS_NUM_THREADS = "2", BGNS_SIMD = "1")
c2 <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
k2 <- bicor_knn(x, knn = 4, pairwise.complete.obs = TRUE,
                threshold = -Inf, min_overlap = 3)
assert_equal(c1, c2, tolerance = 1e-10, msg = "dense bicor differs by thread/SIMD setting")
assert_equal(normalize_knn(k1), normalize_knn(k2), tolerance = 1e-10,
             msg = "dense KNN differs by thread/SIMD setting")
assert_true(identical(dim(c2), c(10L, 10L)), "dense bicor has wrong dimensions")
assert_true(all(abs(diag(c2) - 1) < 1e-12 | is.na(diag(c2))), "dense bicor diagonal is invalid")

# Tidy output must agree with dense values.
td <- bicor(x, tidy = TRUE, threshold = 0.25, min_overlap = 3)
assert_true(all(abs(td$cor) >= 0.25), "tidy threshold semantics are incorrect")
for (ii in seq_len(nrow(td))) {
  assert_equal(td$cor[ii], c2[as.integer(td$col1[ii]), as.integer(td$col2[ii])],
               tolerance = 1e-8, msg = "tidy edge does not match dense bicor")
}

# pairwise.complete.obs = FALSE must exclude incomplete columns.
xf <- x
xf[, c(1, 4, 7)] <- matrix(rnorm(nrow(xf) * 3), nrow = nrow(xf))
complete_cols <- which(colSums(!is.finite(xf)) == 0L)
kn_complete <- bicor_knn(xf, knn = 3, pairwise.complete.obs = FALSE,
                         threshold = -Inf, min_overlap = 3)
assert_true(all(as.integer(kn_complete$col1) %in% complete_cols),
            "incomplete target column escaped complete-observation mode")
assert_true(all(as.integer(kn_complete$col2) %in% complete_cols),
            "incomplete candidate column escaped complete-observation mode")

# Sparse and dense parity, including non-dgC coercion.
set.seed(1202)
xs <- Matrix::rsparsematrix(80, 9, density = 0.35)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
cd <- bicor(as.matrix(xs), pairwise.complete.obs = TRUE, min_overlap = 1)
cs <- bicor(xs, pairwise.complete.obs = TRUE, min_overlap = 1)
assert_equal(cs, cd, tolerance = 1e-8, msg = "sparse bicor differs from dense reference")
ks <- bicor_knn(xs, knn = 3, threshold = -Inf, min_overlap = 1)
kd <- bicor_knn(as.matrix(xs), knn = 3, threshold = -Inf, min_overlap = 1)
assert_equal(normalize_knn(ks), normalize_knn(kd), tolerance = 1e-8,
             msg = "sparse KNN differs from dense reference")
xt <- methods::as(xs, "dgTMatrix")
kt <- bicor_knn(xt, knn = 3, threshold = -Inf, min_overlap = 1)
assert_equal(normalize_knn(kt), normalize_knn(ks), tolerance = 1e-8,
             msg = "coercible sparse class differs from dgCMatrix")

# Mixed sparse/dense bipartite paths in both orientations.
set.seed(1203)
yd <- matrix(rnorm(80 * 4), nrow = 80, ncol = 4)
colnames(yd) <- paste0("y", seq_len(ncol(yd)))
mx1 <- bicor_knn(xs, yd, knn = 3, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
rf1 <- bicor_knn(as.matrix(xs), yd, knn = 3, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
assert_equal(normalize_knn(mx1), normalize_knn(rf1), tolerance = 1e-8,
             msg = "sparse-x/dense-y mixed path differs from dense reference")
ys <- Matrix::rsparsematrix(80, 4, density = 0.4)
colnames(ys) <- paste0("ys", seq_len(ncol(ys)))
xd <- matrix(rnorm(80 * 9), nrow = 80, ncol = 9)
colnames(xd) <- paste0("xd", seq_len(ncol(xd)))
mx2 <- bicor_knn(xd, ys, knn = 3, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
rf2 <- bicor_knn(xd, as.matrix(ys), knn = 3, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
assert_equal(normalize_knn(mx2), normalize_knn(rf2), tolerance = 1e-8,
             msg = "dense-x/sparse-y mixed path differs from dense reference")

# Rectangular dense bicor orientation and KNN graph direction.
rect <- bicor(xd, yd, pairwise.complete.obs = TRUE, min_overlap = 1)
assert_true(identical(dim(rect), c(ncol(xd), ncol(yd))), "rectangular bicor dimensions are wrong")
for (i in seq_len(ncol(xd))) {
  for (j in seq_len(ncol(yd))) {
    ref <- bicor(cbind(xd[, i], yd[, j]), min_overlap = 1)[1, 2]
    assert_equal(rect[i, j], ref, tolerance = 1e-8,
                 msg = "rectangular bicor indexing is wrong")
  }
}

# direct_sparse and dense-return KNN paths should agree.
k_sparse <- bicor_knn(xd, knn = 3, threshold = -Inf, min_overlap = 1,
                      direct_sparse = TRUE)
k_dense <- bicor_knn(xd, knn = 3, threshold = -Inf, min_overlap = 1,
                     direct_sparse = FALSE)
assert_equal(normalize_knn(k_sparse), normalize_knn(k_dense), tolerance = 1e-8,
             msg = "direct_sparse TRUE/FALSE paths differ")

# Edge cases and validation.
constant <- cbind(a = rep(1, 20), b = seq_len(20), c = rep(NA_real_, 20))
cc <- bicor(constant, pairwise.complete.obs = TRUE, min_overlap = 3)
assert_true(identical(dim(cc), c(3L, 3L)), "constant/all-NA input returned wrong shape")
err <- try(bicor_knn(matrix(1:20, 10, 2), matrix(1:18, 9, 2), knn = 1), silent = TRUE)
assert_true(inherits(err, "try-error"), "row-count mismatch was not rejected")

cat("All focused runtime validations passed.\n")
