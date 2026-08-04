mode <- commandArgs(trailingOnly = TRUE)
if (length(mode) != 1L) stop("expected one runtime mode argument")

library(bgns)
library(Matrix)

stopifnot(as.character(packageVersion("bgns")) == "0.4.2")

normalize_knn <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z[order(z$col1, z$rank, z$col2), , drop = FALSE]
}

set.seed(9101)
x <- matrix(rnorm(720), nrow = 80, ncol = 9)
x[cbind(c(2, 7, 15, 31), c(1, 3, 5, 8))] <- NA_real_
colnames(x) <- paste0("x", seq_len(ncol(x)))

# Dense NA-aware bicor and exact KNN parity.
cm <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
stopifnot(identical(dim(cm), c(9L, 9L)))
stopifnot(isTRUE(all.equal(cm, t(cm), tolerance = 1e-12, check.attributes = FALSE)))

kn <- bicor_knn(x, knn = 3, pairwise.complete.obs = TRUE,
                threshold = -Inf, min_overlap = 3)
stopifnot(all(kn$rank %in% 1:3))
for (j in seq_len(ncol(x))) {
  ref <- order(cm[, j], decreasing = TRUE, na.last = NA)
  ref <- ref[ref != j]
  ref <- head(ref, 3L)
  got <- kn[as.integer(kn$col1) == j, , drop = FALSE]
  got <- got[order(got$rank), , drop = FALSE]
  stopifnot(identical(as.integer(got$col2), as.integer(ref)))
  stopifnot(isTRUE(all.equal(got$val, cm[cbind(ref, rep.int(j, length(ref)))], tolerance = 1e-8)))
}

# pairwise.complete.obs = FALSE must exclude incomplete columns.
kn_complete <- bicor_knn(x, knn = 2, pairwise.complete.obs = FALSE,
                         threshold = -Inf, min_overlap = 3)
complete_cols <- which(colSums(!is.finite(x)) == 0L)
stopifnot(all(as.integer(kn_complete$col1) %in% complete_cols))
stopifnot(all(as.integer(kn_complete$col2) %in% complete_cols))

# Sparse and dense paths must agree numerically.
set.seed(9102)
xs <- Matrix::rsparsematrix(75, 8, density = 0.42)
colnames(xs) <- paste0("s", seq_len(ncol(xs)))
cm_sparse <- bicor(xs, pairwise.complete.obs = TRUE, min_overlap = 1)
cm_dense <- bicor(as.matrix(xs), pairwise.complete.obs = TRUE, min_overlap = 1)
stopifnot(isTRUE(all.equal(cm_sparse, cm_dense, tolerance = 1e-8, check.attributes = FALSE)))

kn_sparse <- normalize_knn(bicor_knn(xs, knn = 3, threshold = -Inf, min_overlap = 1))
kn_dense <- normalize_knn(bicor_knn(as.matrix(xs), knn = 3, threshold = -Inf, min_overlap = 1))
stopifnot(isTRUE(all.equal(kn_sparse, kn_dense, tolerance = 1e-8, check.attributes = FALSE)))

# Rectangular dense bicor and both mixed sparse/dense KNN directions.
set.seed(9103)
y <- matrix(rnorm(75 * 4), nrow = 75, ncol = 4)
colnames(y) <- paste0("y", seq_len(ncol(y)))
rect <- bicor(as.matrix(xs), y, pairwise.complete.obs = TRUE, min_overlap = 1)
stopifnot(identical(dim(rect), c(8L, 4L)))

mixed_sd <- normalize_knn(bicor_knn(xs, y, knn = 3, threshold = -Inf,
                                    min_overlap = 1, bipartite_levels = "separate"))
ref_sd <- normalize_knn(bicor_knn(as.matrix(xs), y, knn = 3, threshold = -Inf,
                                  min_overlap = 1, bipartite_levels = "separate"))
stopifnot(isTRUE(all.equal(mixed_sd, ref_sd, tolerance = 1e-8, check.attributes = FALSE)))

ys <- Matrix::Matrix(y, sparse = TRUE)
mixed_ds <- normalize_knn(bicor_knn(as.matrix(xs), ys, knn = 3, threshold = -Inf,
                                    min_overlap = 1, bipartite_levels = "separate"))
ref_ds <- normalize_knn(bicor_knn(as.matrix(xs), as.matrix(ys), knn = 3, threshold = -Inf,
                                  min_overlap = 1, bipartite_levels = "separate"))
stopifnot(isTRUE(all.equal(mixed_ds, ref_ds, tolerance = 1e-8, check.attributes = FALSE)))

cat(sprintf("runtime validation passed: mode=%s, OS=%s, R=%s\n",
            mode, Sys.info()[["sysname"]], getRversion()))
