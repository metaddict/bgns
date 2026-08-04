args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: runtime-mode.R <library> <output.rds>")
lib <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
out <- args[[2L]]
.libPaths(c(lib, .libPaths()))

suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

assert <- function(ok, msg) {
  if (!isTRUE(ok)) stop(msg, call. = FALSE)
}
assert_equal <- function(x, y, tolerance = 1e-9, msg = "objects differ") {
  z <- all.equal(x, y, tolerance = tolerance, check.attributes = TRUE)
  if (!isTRUE(z)) stop(sprintf("%s: %s", msg, paste(z, collapse = "; ")), call. = FALSE)
}
expect_error <- function(expr, pattern) {
  err <- tryCatch({ force(expr); NULL }, error = identity)
  assert(inherits(err, "error"), sprintf("expected error matching %s", pattern))
  assert(grepl(pattern, conditionMessage(err), fixed = FALSE),
         sprintf("error did not match %s: %s", pattern, conditionMessage(err)))
}

normalize_knn <- function(z) {
  z <- data.frame(
    col1 = as.integer(z$col1),
    col2 = as.integer(z$col2),
    val = as.numeric(z$val),
    rank = as.integer(z$rank)
  )
  z[order(z$col1, z$rank, z$col2, -z$val), , drop = FALSE]
}
canonical_tidy <- function(z) {
  if (!nrow(z)) return(data.frame(pair = character(), cor = numeric()))
  a <- as.integer(z$col1)
  b <- as.integer(z$col2)
  d <- data.frame(pair = paste(pmin(a, b), pmax(a, b), sep = "-"), cor = as.numeric(z$cor))
  d <- aggregate(cor ~ pair, data = d, FUN = mean)
  d[order(d$pair), , drop = FALSE]
}

assert(identical(as.character(packageVersion("bgns")), "0.4.2"), "wrong installed bgns version")

set.seed(4201)
x <- matrix(rnorm(240L * 18L), nrow = 240L, ncol = 18L)
colnames(x) <- paste0("g", seq_len(ncol(x)))
x[sample.int(length(x), 100L)] <- NA_real_

cm <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3L)
assert(identical(dim(cm), c(18L, 18L)), "dense bicor dimensions are wrong")
assert(max(abs(cm - t(cm)), na.rm = TRUE) < 1e-10, "dense bicor is not symmetric")
assert(all(abs(diag(cm) - 1) < 1e-10, na.rm = TRUE), "dense bicor diagonal is not one")

# Rectangular indexing and bipartite orientation.
set.seed(4202)
xr <- matrix(rnorm(96L), nrow = 16L, ncol = 6L)
yr <- matrix(rnorm(64L), nrow = 16L, ncol = 4L)
rect <- bicor(xr, yr, min_overlap = 1L)
assert(identical(dim(rect), c(6L, 4L)), "rectangular bicor dimensions are wrong")
for (i in seq_len(ncol(xr))) for (j in seq_len(ncol(yr))) {
  ref <- bicor(cbind(xr[, i], yr[, j]), min_overlap = 1L)[1L, 2L]
  assert_equal(rect[i, j], ref, tolerance = 1e-10, msg = "rectangular bicor indexing mismatch")
}

# Streaming top-k and dense-intermediate route must agree exactly up to rounding.
kd <- bicor_knn(x, knn = 5L, threshold = -Inf, min_overlap = 3L, direct_sparse = TRUE)
kf <- bicor_knn(x, knn = 5L, threshold = -Inf, min_overlap = 3L, direct_sparse = FALSE)
assert_equal(normalize_knn(kd), normalize_knn(kf), tolerance = 1e-9,
             msg = "direct_sparse TRUE/FALSE mismatch")
assert(!any(as.integer(kd$col1) == as.integer(kd$col2)), "self edges found in KNN output")
assert(all(kd$rank >= 1L & kd$rank <= 5L), "invalid KNN ranks")
assert(all(table(as.integer(kd$col1)) <= 5L), "too many KNN edges per target")

# Signed threshold semantics.
kpos <- bicor_knn(x, knn = 5L, threshold = 0, min_overlap = 3L)
assert(all(kpos$val > 0), "threshold = 0 retained a non-positive edge")

# pairwise.complete.obs = FALSE must exclude every incomplete column.
x_false <- x
x_false[, c(1L, 3L, 7L)] <- matrix(rnorm(nrow(x_false) * 3L), nrow = nrow(x_false))
x_false[1L, 2L] <- NA_real_
x_false[2L, 5L] <- Inf
complete_cols <- which(colSums(!is.finite(x_false)) == 0L)
kfalse <- bicor_knn(x_false, knn = 3L, pairwise.complete.obs = FALSE,
                    threshold = -Inf, min_overlap = 1L)
assert(all(as.integer(kfalse$col1) %in% complete_cols), "incomplete target retained with pairwise FALSE")
assert(all(as.integer(kfalse$col2) %in% complete_cols), "incomplete neighbor retained with pairwise FALSE")

# Sparse and dense representations must agree for structural-zero numeric matrices.
set.seed(4203)
xs <- Matrix::rsparsematrix(120L, 14L, density = 0.28)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
t_sparse <- bicor(xs, tidy = TRUE, threshold = -Inf, min_overlap = 1L)
t_dense <- bicor(as.matrix(xs), tidy = TRUE, threshold = -Inf, min_overlap = 1L)
assert_equal(canonical_tidy(t_sparse), canonical_tidy(t_dense), tolerance = 1e-8,
             msg = "sparse/dense tidy bicor mismatch")
ks <- bicor_knn(xs, knn = 4L, threshold = -Inf, min_overlap = 1L)
ks_ref <- bicor_knn(as.matrix(xs), knn = 4L, threshold = -Inf, min_overlap = 1L)
assert_equal(normalize_knn(ks), normalize_knn(ks_ref), tolerance = 1e-8,
             msg = "sparse/dense KNN mismatch")

# User-facing sparse coercion should accept another Matrix sparse representation.
xt <- methods::as(xs, "TsparseMatrix")
kt <- bicor_knn(xt, knn = 3L, threshold = -Inf, min_overlap = 1L)
kt_ref <- bicor_knn(xs, knn = 3L, threshold = -Inf, min_overlap = 1L)
assert_equal(normalize_knn(kt), normalize_knn(kt_ref), tolerance = 1e-8,
             msg = "TsparseMatrix coercion mismatch")

# Mixed sparse/dense bipartite paths must match dense references in both directions.
set.seed(4204)
yd <- matrix(rnorm(120L * 6L), nrow = 120L, ncol = 6L)
colnames(yd) <- paste0("y", seq_len(ncol(yd)))
km1 <- bicor_knn(xs, yd, knn = 3L, threshold = -Inf, min_overlap = 1L,
                 bipartite_levels = "separate")
km1_ref <- bicor_knn(as.matrix(xs), yd, knn = 3L, threshold = -Inf, min_overlap = 1L,
                     bipartite_levels = "separate")
assert_equal(normalize_knn(km1), normalize_knn(km1_ref), tolerance = 1e-8,
             msg = "sparse-x/dense-y KNN mismatch")

set.seed(4205)
xd <- matrix(rnorm(120L * 11L), nrow = 120L, ncol = 11L)
colnames(xd) <- paste0("x", seq_len(ncol(xd)))
ys <- Matrix::rsparsematrix(120L, 5L, density = 0.33)
colnames(ys) <- paste0("z", seq_len(ncol(ys)))
km2 <- bicor_knn(xd, ys, knn = 3L, threshold = -Inf, min_overlap = 1L,
                 bipartite_levels = "separate")
km2_ref <- bicor_knn(xd, as.matrix(ys), knn = 3L, threshold = -Inf, min_overlap = 1L,
                     bipartite_levels = "separate")
assert_equal(normalize_knn(km2), normalize_knn(km2_ref), tolerance = 1e-8,
             msg = "dense-x/sparse-y KNN mismatch")

# Strict versus separate bipartite levels.
expect_error(bicor_knn(xd, as.matrix(ys), knn = 2L), "bipartite")
assert(is.factor(km2$col1) && is.factor(km2$col2), "separate bipartite levels are not factors")
assert(identical(levels(km2$col1), colnames(ys)), "target factor levels are wrong")
assert(identical(levels(km2$col2), colnames(xd)), "source factor levels are wrong")

# Edge cases and validation.
small <- matrix(rnorm(60L), nrow = 20L, ncol = 3L)
kbig <- bicor_knn(small, knn = 20L, threshold = -Inf, min_overlap = 1L)
assert(all(table(as.integer(kbig$col1)) <= 2L), "knn larger than candidate count mishandled")
expect_error(bicor(x, min_overlap = 0L), "min_overlap")
expect_error(bicor_knn(x, knn = 0L), "knn")
expect_error(bicor(xr, yr[-1L, , drop = FALSE]), "rows|row")

# Save canonical outputs so separate processes can compare SIMD/thread modes.
result <- list(
  dense = cm,
  rectangular = rect,
  dense_knn = normalize_knn(kd),
  sparse_tidy = canonical_tidy(t_sparse),
  sparse_knn = normalize_knn(ks),
  mixed_forward = normalize_knn(km1),
  mixed_reverse = normalize_knn(km2),
  pairwise_false = normalize_knn(kfalse)
)
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
saveRDS(result, out, version = 3L)
cat(sprintf("runtime validation passed: %s; BGNS_SIMD=%s; BGNS_NUM_THREADS=%s\n",
            R.version.string,
            Sys.getenv("BGNS_SIMD", unset = "<unset>"),
            Sys.getenv("BGNS_NUM_THREADS", unset = "<unset>")))
