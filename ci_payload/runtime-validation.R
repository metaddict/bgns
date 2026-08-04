suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

stopifnot(as.character(utils::packageVersion("bgns")) == "0.4.2")

normalize_tidy <- function(z, value = "cor") {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z <- z[order(z$col1, z$col2), c("col1", "col2", value), drop = FALSE]
  rownames(z) <- NULL
  z
}

normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z <- z[order(z$col1, z$rank, z$col2), c("col1", "col2", "val", "rank"), drop = FALSE]
  rownames(z) <- NULL
  z
}

expect_equal_num <- function(x, y, tolerance = 1e-9, label = "objects") {
  ans <- all.equal(x, y, tolerance = tolerance, check.attributes = TRUE)
  if (!isTRUE(ans)) {
    stop(sprintf("%s differ: %s", label, paste(ans, collapse = "; ")), call. = FALSE)
  }
}

with_engine <- function(simd, threads, expr) {
  old_simd <- Sys.getenv("BGNS_SIMD", unset = NA_character_)
  old_threads <- Sys.getenv("BGNS_NUM_THREADS", unset = NA_character_)
  on.exit({
    if (is.na(old_simd)) Sys.unsetenv("BGNS_SIMD") else Sys.setenv(BGNS_SIMD = old_simd)
    if (is.na(old_threads)) Sys.unsetenv("BGNS_NUM_THREADS") else Sys.setenv(BGNS_NUM_THREADS = old_threads)
  }, add = TRUE)
  Sys.setenv(BGNS_SIMD = simd, BGNS_NUM_THREADS = as.character(threads))
  force(expr)
}

message("[1/8] dense bicor, NA paths, rectangular x/y")
set.seed(4201)
x <- matrix(rnorm(240L * 48L), nrow = 240L, ncol = 48L)
x[sample(length(x), 180L)] <- NA_real_
colnames(x) <- sprintf("x%02d", seq_len(ncol(x)))

y <- matrix(rnorm(240L * 11L), nrow = 240L, ncol = 11L)
y[sample(length(y), 35L)] <- NA_real_
colnames(y) <- sprintf("y%02d", seq_len(ncol(y)))

cm <- bicor(x, min_overlap = 3)
stopifnot(is.matrix(cm), identical(dim(cm), c(48L, 48L)))
stopifnot(isTRUE(all.equal(cm, t(cm), tolerance = 1e-10, check.attributes = FALSE)))
xy <- bicor(x, y, min_overlap = 3)
stopifnot(identical(dim(xy), c(48L, 11L)))
for (ij in list(c(1L, 1L), c(7L, 3L), c(48L, 11L))) {
  ref <- bicor(cbind(x[, ij[1L]], y[, ij[2L]]), min_overlap = 3)[1L, 2L]
  expect_equal_num(xy[ij[1L], ij[2L]], ref, 1e-10, "rectangular dense bicor")
}

message("[2/8] scalar/SIMD and one/two-thread numerical consistency")
scalar_cm <- with_engine("0", 1L, bicor(x, min_overlap = 3))
parallel_cm <- with_engine("1", 2L, bicor(x, min_overlap = 3))
expect_equal_num(scalar_cm, parallel_cm, 1e-9, "dense scalar/parallel bicor")

scalar_kn <- with_engine("0", 1L, bicor_knn(x, knn = 7L, threshold = -Inf, min_overlap = 3))
parallel_kn <- with_engine("1", 2L, bicor_knn(x, knn = 7L, threshold = -Inf, min_overlap = 3))
expect_equal_num(normalize_knn(scalar_kn), normalize_knn(parallel_kn), 1e-9,
                 "dense scalar/parallel KNN")

message("[3/8] streamed versus dense-intermediate KNN")
stream_kn <- bicor_knn(x, knn = 6L, direct_sparse = TRUE, threshold = -Inf, min_overlap = 3)
dense_kn <- bicor_knn(x, knn = 6L, direct_sparse = FALSE, threshold = -Inf, min_overlap = 3)
expect_equal_num(normalize_knn(stream_kn), normalize_knn(dense_kn), 1e-9,
                 "streamed/dense-intermediate KNN")

message("[4/8] sparse tidy and sparse KNN parity")
set.seed(4202)
xs <- Matrix::rsparsematrix(180L, 37L, density = 0.18)
colnames(xs) <- sprintf("s%02d", seq_len(ncol(xs)))
sp_tidy <- bicor(xs, tidy = TRUE, threshold = -Inf, min_overlap = 1)
de_tidy <- bicor(as.matrix(xs), tidy = TRUE, threshold = -Inf, min_overlap = 1)
expect_equal_num(normalize_tidy(sp_tidy), normalize_tidy(de_tidy), 1e-8,
                 "sparse/dense tidy bicor")

sp_kn <- bicor_knn(xs, knn = 5L, threshold = -Inf, min_overlap = 1)
de_kn <- bicor_knn(as.matrix(xs), knn = 5L, threshold = -Inf, min_overlap = 1)
expect_equal_num(normalize_knn(sp_kn), normalize_knn(de_kn), 1e-8,
                 "sparse/dense KNN")

message("[5/8] sparse-class coercion")
xt <- methods::as(xs, "TsparseMatrix")
stopifnot(inherits(xt, "sparseMatrix"), !inherits(xt, "dgCMatrix"))
coerced_kn <- bicor_knn(xt, knn = 4L, threshold = -Inf, min_overlap = 1)
base_kn <- bicor_knn(xs, knn = 4L, threshold = -Inf, min_overlap = 1)
expect_equal_num(normalize_knn(coerced_kn), normalize_knn(base_kn), 1e-8,
                 "non-dgC sparse coercion")

message("[6/8] mixed sparse/dense bipartite KNN in both orientations")
set.seed(4203)
yd <- matrix(rnorm(180L * 9L), nrow = 180L, ncol = 9L)
colnames(yd) <- sprintf("yd%02d", seq_len(ncol(yd)))
mx1 <- bicor_knn(xs, yd, knn = 4L, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
rf1 <- bicor_knn(as.matrix(xs), yd, knn = 4L, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
expect_equal_num(normalize_knn(mx1), normalize_knn(rf1), 1e-8,
                 "sparse-x/dense-y mixed KNN")

ys <- Matrix::rsparsematrix(180L, 9L, density = 0.22)
xd <- matrix(rnorm(180L * 37L), nrow = 180L, ncol = 37L)
colnames(ys) <- sprintf("ys%02d", seq_len(ncol(ys)))
colnames(xd) <- sprintf("xd%02d", seq_len(ncol(xd)))
mx2 <- bicor_knn(xd, ys, knn = 4L, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
rf2 <- bicor_knn(xd, as.matrix(ys), knn = 4L, threshold = -Inf, min_overlap = 1,
                 bipartite_levels = "separate")
expect_equal_num(normalize_knn(mx2), normalize_knn(rf2), 1e-8,
                 "dense-x/sparse-y mixed KNN")

message("[7/8] complete-column semantics, thresholds, and deterministic ties")
x_missing <- x[, 1:12, drop = FALSE]
x_missing[1L, c(2L, 6L)] <- NA_real_
complete_cols <- which(colSums(!is.finite(x_missing)) == 0L)
kn_complete <- bicor_knn(x_missing, knn = 3L, pairwise.complete.obs = FALSE,
                         threshold = -Inf, min_overlap = 1)
stopifnot(all(as.integer(kn_complete$col1) %in% complete_cols))
stopifnot(all(as.integer(kn_complete$col2) %in% complete_cols))
kn_positive <- bicor_knn(x[, 1:15, drop = FALSE], knn = 4L, threshold = 0, min_overlap = 3)
stopifnot(all(kn_positive$val > 0))

set.seed(4204)
v <- rnorm(120L)
ties <- cbind(v, v, v, -v, rnorm(120L), rnorm(120L))
tie1 <- bicor_knn(ties, knn = 3L, threshold = -Inf, min_overlap = 3)
tie2 <- bicor_knn(ties, knn = 3L, threshold = -Inf, min_overlap = 3)
stopifnot(identical(tie1, tie2))

message("[8/8] repeated deterministic stress loop")
for (seed in 1:8) {
  set.seed(5000L + seed)
  z <- matrix(rnorm(96L * 28L), nrow = 96L, ncol = 28L)
  z[sample(length(z), 25L)] <- NA_real_
  one <- with_engine("0", 1L, bicor_knn(z, knn = 5L, threshold = -Inf, min_overlap = 2))
  two <- with_engine("1", 2L, bicor_knn(z, knn = 5L, threshold = -Inf, min_overlap = 2))
  expect_equal_num(normalize_knn(one), normalize_knn(two), 1e-9,
                   sprintf("stress iteration %d", seed))
}

message("Runtime validation completed successfully on ", R.version.string,
        " / ", Sys.info()[["sysname"]], " ", Sys.info()[["machine"]])
