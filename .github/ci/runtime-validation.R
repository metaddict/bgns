options(warn = 2)

library(bgns)
library(Matrix)

stopifnot(as.character(utils::packageVersion("bgns")) == "0.4.2")

assert_close <- function(actual, expected, tolerance = 1e-8, label = "values") {
  ok <- isTRUE(all.equal(actual, expected, tolerance = tolerance, check.attributes = TRUE))
  if (!ok) {
    stop(label, " differ:\n", paste(capture.output(all.equal(actual, expected, tolerance = tolerance)), collapse = "\n"))
  }
}

midvar_ref <- function(x) {
  good <- is.finite(x)
  vals <- x[good]
  if (!length(vals)) return(list(mv = rep(NA_real_, length(x)), ss = NA_real_, good = good, ok = FALSE))
  med <- stats::median(vals)
  mad <- 1.4826 * stats::median(abs(vals - med))
  if (!is.finite(mad) || mad <= 0) return(list(mv = rep(NA_real_, length(x)), ss = NA_real_, good = good, ok = FALSE))
  u <- (vals - med) / (9 * mad)
  w <- ifelse(abs(u) >= 1, 0, (1 - u^2)^2)
  mv_good <- (vals - med) * w
  mv <- rep(NA_real_, length(x))
  mv[good] <- mv_good
  ss <- sum(mv_good^2)
  list(mv = mv, ss = ss, good = good, ok = is.finite(ss) && ss > 0)
}

bicor_pair_ref <- function(x, y, pairwise = TRUE, intersection_denominator = FALSE, min_overlap = 3L) {
  if (!pairwise && (!all(is.finite(x)) || !all(is.finite(y)))) return(NA_real_)
  px <- midvar_ref(x)
  py <- midvar_ref(y)
  if (!px$ok || !py$ok) return(NA_real_)
  overlap <- px$good & py$good
  if (sum(overlap) < min_overlap) return(NA_real_)
  num <- sum(px$mv[overlap] * py$mv[overlap])
  if (intersection_denominator) {
    den <- sqrt(sum(px$mv[overlap]^2) * sum(py$mv[overlap]^2))
  } else {
    den <- sqrt(px$ss * py$ss)
  }
  if (!is.finite(den) || den <= 0) return(NA_real_)
  num / den
}

bicor_ref <- function(x, y = NULL, pairwise = TRUE, intersection_denominator = FALSE, min_overlap = 3L) {
  x <- as.matrix(x)
  y <- if (is.null(y)) x else as.matrix(y)
  out <- matrix(NA_real_, ncol(x), ncol(y), dimnames = list(colnames(x), colnames(y)))
  for (i in seq_len(ncol(x))) {
    for (j in seq_len(ncol(y))) {
      out[i, j] <- bicor_pair_ref(
        x[, i], y[, j], pairwise = pairwise,
        intersection_denominator = intersection_denominator,
        min_overlap = min_overlap
      )
    }
  }
  out
}

normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z[order(z$col1, z$rank, z$col2), , drop = FALSE]
}

set.seed(1001)
x <- matrix(rnorm(800), nrow = 80, ncol = 10)
colnames(x) <- paste0("x", seq_len(ncol(x)))
actual <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
expected <- bicor_ref(x, pairwise = TRUE, min_overlap = 3)
assert_close(actual, expected, tolerance = 1e-10, label = "dense bicor")

x_na <- x
x_na[sample(length(x_na), 37)] <- NA_real_
actual_na <- bicor(x_na, pairwise.complete.obs = TRUE, min_overlap = 3)
expected_na <- bicor_ref(x_na, pairwise = TRUE, min_overlap = 3)
assert_close(actual_na, expected_na, tolerance = 1e-10, label = "pairwise-NA bicor")

actual_inter <- bicor(
  x_na, pairwise.complete.obs = TRUE,
  use_intersection_denominator = TRUE, min_overlap = 3
)
expected_inter <- bicor_ref(
  x_na, pairwise = TRUE, intersection_denominator = TRUE, min_overlap = 3
)
assert_close(actual_inter, expected_inter, tolerance = 1e-10, label = "intersection-denominator bicor")

actual_complete <- bicor(x_na, pairwise.complete.obs = FALSE, min_overlap = 3)
expected_complete <- bicor_ref(x_na, pairwise = FALSE, min_overlap = 3)
assert_close(actual_complete, expected_complete, tolerance = 1e-10, label = "complete-column bicor")

set.seed(1002)
y <- matrix(rnorm(80 * 4), nrow = 80, ncol = 4)
y[sample(length(y), 11)] <- NA_real_
colnames(y) <- paste0("y", seq_len(ncol(y)))
actual_xy <- bicor(x_na, y, pairwise.complete.obs = TRUE, min_overlap = 3)
expected_xy <- bicor_ref(x_na, y, pairwise = TRUE, min_overlap = 3)
assert_close(actual_xy, expected_xy, tolerance = 1e-10, label = "rectangular bicor")

set.seed(1003)
xs <- Matrix::rsparsematrix(90, 12, density = 0.35)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
sparse_bicor <- bicor(xs, pairwise.complete.obs = TRUE, min_overlap = 3)
dense_sparse_bicor <- bicor(as.matrix(xs), pairwise.complete.obs = TRUE, min_overlap = 3)
assert_close(sparse_bicor, dense_sparse_bicor, tolerance = 1e-10, label = "sparse/dense bicor parity")

kn <- bicor_knn(x, knn = 4, threshold = -Inf, min_overlap = 3)
cm <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
for (j in seq_len(ncol(x))) {
  ord <- order(cm[, j], decreasing = TRUE, na.last = NA)
  ord <- ord[ord != j]
  ord <- ord[seq_len(min(4L, length(ord)))]
  got <- kn[as.integer(kn$col1) == j, , drop = FALSE]
  got <- got[order(got$rank, as.integer(got$col2)), , drop = FALSE]
  stopifnot(identical(as.integer(got$col2), as.integer(ord)))
  assert_close(got$val, cm[ord, j], tolerance = 1e-10, label = paste("KNN target", j))
}

set.seed(1004)
y_dense <- matrix(rnorm(90 * 5), nrow = 90, ncol = 5)
colnames(y_dense) <- paste0("d", seq_len(ncol(y_dense)))
mixed1 <- bicor_knn(xs, y_dense, knn = 3, threshold = -Inf, min_overlap = 3,
                    bipartite_levels = "separate")
ref1 <- bicor_knn(as.matrix(xs), y_dense, knn = 3, threshold = -Inf, min_overlap = 3,
                  bipartite_levels = "separate")
assert_close(normalize_knn(mixed1), normalize_knn(ref1), tolerance = 1e-10,
             label = "sparse/dense KNN parity")

ys <- Matrix::rsparsematrix(90, 5, density = 0.4)
colnames(ys) <- paste0("ys", seq_len(ncol(ys)))
x_dense <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
colnames(x_dense) <- paste0("dx", seq_len(ncol(x_dense)))
mixed2 <- bicor_knn(x_dense, ys, knn = 3, threshold = -Inf, min_overlap = 3,
                    bipartite_levels = "separate")
ref2 <- bicor_knn(x_dense, as.matrix(ys), knn = 3, threshold = -Inf, min_overlap = 3,
                  bipartite_levels = "separate")
assert_close(normalize_knn(mixed2), normalize_knn(ref2), tolerance = 1e-10,
             label = "dense/sparse KNN parity")

Sys.setenv(BGNS_NUM_THREADS = "1")
one_thread <- bicor_knn(x, knn = 4, threshold = -Inf, min_overlap = 3)
Sys.setenv(BGNS_NUM_THREADS = "2")
two_threads <- bicor_knn(x, knn = 4, threshold = -Inf, min_overlap = 3)
assert_close(normalize_knn(one_thread), normalize_knn(two_threads), tolerance = 0,
             label = "thread-count determinism")

testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = TRUE)

cat("Runtime validation completed successfully on ", R.version.string, " / ", Sys.info()[["sysname"]], "\n", sep = "")
