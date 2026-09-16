suppressPackageStartupMessages({
  library(bgns)
  library(Matrix)
})

stopifnot(as.character(utils::packageVersion("bgns")) == "0.4.2")

# Dense, rectangular, tidy and KNN smoke tests.
set.seed(42)
x <- matrix(rnorm(240), nrow = 40, ncol = 6)
y <- matrix(rnorm(120), nrow = 40, ncol = 3)
x[1, 2] <- NA_real_

cm <- bicor(x, min_overlap = 3)
stopifnot(identical(dim(cm), c(6L, 6L)))
xy <- bicor(x, y, min_overlap = 3)
stopifnot(identical(dim(xy), c(6L, 3L)))

td <- bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 3)
stopifnot(identical(names(td), c("col1", "col2", "cor")))

kn <- bicor_knn(x, knn = 2, threshold = -Inf, min_overlap = 3)
stopifnot(identical(names(kn), c("col1", "col2", "val", "rank")))
stopifnot(all(kn$rank %in% 1:2))

# Undefined self-correlations remain NA.
z <- cbind(
  valid = 1:6,
  constant = rep(2, 6),
  incomplete = c(1:5, NA_real_),
  all_missing = rep(NA_real_, 6)
)
cz_pair <- bicor(z, pairwise.complete.obs = TRUE, min_overlap = 3)
stopifnot(identical(cz_pair[1, 1], 1))
stopifnot(is.na(cz_pair[2, 2]))
stopifnot(identical(cz_pair[3, 3], 1))
stopifnot(is.na(cz_pair[4, 4]))

cz_complete <- bicor(z, pairwise.complete.obs = FALSE, min_overlap = 3)
stopifnot(is.na(cz_complete[2, 2]))
stopifnot(is.na(cz_complete[3, 3]))
stopifnot(is.na(cz_complete[4, 4]))

# min_overlap applies consistently, including fully finite fast paths.
small <- matrix(seq_len(16), nrow = 4, ncol = 4)
small <- sweep(small, 2, c(0, 2, -1, 3), "+")
stopifnot(all(is.na(bicor(small, min_overlap = 5))))
stopifnot(nrow(bicor(small, tidy = TRUE, threshold = -Inf, min_overlap = 5)) == 0L)
stopifnot(nrow(bicor_knn(small, knn = 2, threshold = -Inf, min_overlap = 5)) == 0L)

# Non-integral values are rejected instead of silently truncated.
stopifnot(inherits(try(bicor(x, min_overlap = 2.5), silent = TRUE), "try-error"))
stopifnot(inherits(try(bicor_knn(x, knn = 2.5), silent = TRUE), "try-error"))

# Triplet sparse matrices should coerce without the Matrix deprecation warning.
xt <- Matrix::sparseMatrix(
  i = c(1, 2, 4, 6, 1, 3, 5),
  j = c(1, 1, 1, 1, 2, 2, 2),
  x = c(1, 2, 3, 4, 5, 6, 7),
  dims = c(6, 3),
  repr = "T"
)
warn <- character()
tdt <- withCallingHandlers(
  bicor(xt, tidy = TRUE, threshold = -Inf, min_overlap = 1),
  warning = function(w) {
    warn <<- c(warn, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
stopifnot(is.data.frame(tdt))
stopifnot(!any(grepl("deprecated", warn, ignore.case = TRUE)))

# Sparse and mixed-input KNN smoke tests.
set.seed(99)
xs <- Matrix::rsparsematrix(30, 5, density = 0.35)
yd <- matrix(rnorm(90), nrow = 30, ncol = 3)
colnames(xs) <- paste0("x", 1:5)
colnames(yd) <- paste0("y", 1:3)
ks <- bicor_knn(xs, knn = 2, threshold = -Inf, min_overlap = 1)
km <- bicor_knn(xs, yd, knn = 2, threshold = -Inf, min_overlap = 1,
                bipartite_levels = "separate")
stopifnot(is.data.frame(ks), is.data.frame(km))

cat("BGNS_RUNTIME_VALIDATION_PASSED\n")

# NA-aware sparse self-KNN must agree with the same matrix represented densely.
normalize_knn <- function(z) {
  out <- data.frame(
    col1 = as.integer(z$col1),
    col2 = as.integer(z$col2),
    val = z$val,
    rank = z$rank
  )
  out[order(out$col1, out$rank, out$col2), , drop = FALSE]
}

set.seed(2026)
dself <- matrix(rnorm(120), nrow = 24, ncol = 5)
dself[abs(dself) < 0.7] <- 0
colnames(dself) <- paste0("v", seq_len(ncol(dself)))
sself <- methods::as(Matrix::Matrix(dself, sparse = TRUE), "CsparseMatrix")
kdself <- bicor_knn(dself, knn = 2, threshold = -Inf, min_overlap = 3,
                    use_intersection_denominator = TRUE)
ksself <- bicor_knn(sself, knn = 2, threshold = -Inf, min_overlap = 3,
                    use_intersection_denominator = TRUE)
stopifnot(isTRUE(all.equal(normalize_knn(ksself), normalize_knn(kdself),
                           tolerance = 1e-12)))

dself_na <- dself
dself_na[2, 2] <- NA_real_
dself_na[7, 4] <- NA_real_
sself_na <- methods::as(Matrix::Matrix(dself_na, sparse = TRUE), "CsparseMatrix")
kdself_na <- bicor_knn(dself_na, knn = 2, threshold = -Inf, min_overlap = 3,
                       pairwise.complete.obs = TRUE,
                       use_intersection_denominator = TRUE)
ksself_na <- bicor_knn(sself_na, knn = 2, threshold = -Inf, min_overlap = 3,
                       pairwise.complete.obs = TRUE,
                       use_intersection_denominator = TRUE)
stopifnot(isTRUE(all.equal(normalize_knn(ksself_na), normalize_knn(kdself_na),
                           tolerance = 1e-12)))

cat("BGNS_SPARSE_SELF_KNN_VALIDATION_PASSED\n")
