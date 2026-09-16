test_that("bicor tidy matches dense lower triangle", {
  set.seed(101)
  x <- matrix(rnorm(200), nrow = 20, ncol = 10)
  td <- bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 1)
  cm <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 1)

  pairs <- combn(ncol(x), 2)
  exp_df <- data.frame(
    pair = apply(pairs, 2, function(v) paste(v[1], v[2], sep = "-")),
    cor = cm[cbind(pairs[1, ], pairs[2, ])]
  )

  td_pairs <- data.frame(
    pair = apply(cbind(pmin(td$col1, td$col2), pmax(td$col1, td$col2)), 1, paste, collapse = "-"),
    cor = td$cor
  )
  td_aggr <- aggregate(cor ~ pair, data = td_pairs, FUN = mean)

  expect_setequal(td_aggr$pair, exp_df$pair)
  td_aligned <- td_aggr[match(exp_df$pair, td_aggr$pair), ]
  expect_equal(td_aligned$cor, exp_df$cor, tolerance = 1e-6)
})

test_that("bicor tidy sparse matches dense reference with low min_overlap", {
  skip_if_not_installed("Matrix")
  set.seed(202)
  xs <- Matrix::rsparsematrix(30, 8, density = 0.7)
  td <- bicor(xs, tidy = TRUE, threshold = -Inf, min_overlap = 1)
  cm <- bicor(as.matrix(xs), pairwise.complete.obs = TRUE, min_overlap = 1)

  pairs <- combn(ncol(xs), 2)
  exp_df <- data.frame(
    pair = apply(pairs, 2, function(v) paste(v[1], v[2], sep = "-")),
    cor = cm[cbind(pairs[1, ], pairs[2, ])]
  )

  td_pairs <- data.frame(
    pair = apply(cbind(pmin(td$col1, td$col2), pmax(td$col1, td$col2)), 1, paste, collapse = "-"),
    cor = td$cor
  )
  td_aggr <- aggregate(cor ~ pair, data = td_pairs, FUN = mean)

  expect_setequal(td_aggr$pair, exp_df$pair)
  td_aligned <- td_aggr[match(exp_df$pair, td_aggr$pair), ]
  expect_equal(td_aligned$cor, exp_df$cor, tolerance = 1e-6)
})

test_that("min_overlap filters pairs as expected", {
  x <- matrix(
    c(1, NA, 2, NA,
      1,  2, NA, NA,
      1, NA, NA,  3,
      NA, 2, NA,  3,
      1,  2, NA, NA),
    nrow = 5,
    ncol = 4
  )
  td_strict <- bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 3)
  td_loose <- bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 1)
  expect_equal(nrow(td_strict), 0L)
  expect_true(nrow(td_loose) > 0L)
})

test_that("bicor_knn ranks and values align with dense bicor", {
  set.seed(303)
  x <- matrix(rnorm(150), nrow = 30, ncol = 5)
  kn <- bicor_knn(x, knn = 2, threshold = -Inf, min_overlap = 1)
  expect_true(all(kn$rank %in% 1:2))

  cm <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 1)
  for (j in seq_len(ncol(x))) {
    ref_order <- order(cm[, j], decreasing = TRUE)
    ref_order <- ref_order[ref_order != j]
    expect_idx <- ref_order[seq_len(2)]
    kn_sub <- kn[kn$col1 == j, ]
    kn_sub <- kn_sub[order(kn_sub$rank), ]
    expect_equal(kn_sub$col2, expect_idx)
    expect_equal(kn_sub$val, cm[expect_idx, j], tolerance = 1e-6)
  }
})

test_that("bicor_knn honors pairwise.complete.obs = FALSE", {
  set.seed(404)
  x <- matrix(rnorm(120), nrow = 20, ncol = 6)
  x[1, 2] <- NA_real_
  x[2, 5] <- NA_real_

  complete_cols <- which(colSums(!is.finite(x)) == 0)
  kn <- bicor_knn(x, knn = 2, pairwise.complete.obs = FALSE, threshold = -Inf, min_overlap = 1)

  expect_true(all(as.integer(kn$col1) %in% complete_cols))
  expect_true(all(as.integer(kn$col2) %in% complete_cols))
  expect_false(any(as.integer(kn$col1) %in% c(2L, 5L)))
  expect_false(any(as.integer(kn$col2) %in% c(2L, 5L)))

  cm <- bicor(x, pairwise.complete.obs = FALSE, min_overlap = 1)
  for (j in complete_cols) {
    ref_order <- order(cm[, j], decreasing = TRUE, na.last = NA)
    ref_order <- ref_order[ref_order != j]
    expect_idx <- ref_order[seq_len(min(2, length(ref_order)))]
    kn_sub <- kn[as.integer(kn$col1) == j, ]
    kn_sub <- kn_sub[order(kn_sub$rank), ]
    expect_equal(as.integer(kn_sub$col2), expect_idx)
    expect_equal(kn_sub$val, cm[expect_idx, j], tolerance = 1e-6)
  }
})

test_that("bicor_knn handles knn larger than available neighbors", {
  set.seed(505)
  x <- matrix(rnorm(80), nrow = 20, ncol = 4)
  kn <- bicor_knn(x, knn = 10, threshold = -Inf, min_overlap = 1)
  expect_true(all(table(kn$col1) <= 3L))
})

test_that("bicor_knn supports bipartite factor levels", {
  set.seed(606)
  x <- matrix(rnorm(100), nrow = 20, ncol = 5)
  y <- matrix(rnorm(60), nrow = 20, ncol = 3)
  colnames(x) <- paste0("x", seq_len(ncol(x)))
  colnames(y) <- paste0("y", seq_len(ncol(y)))

  expect_error(bicor_knn(x, y, knn = 2), "bipartite")
  kn <- bicor_knn(x, y, knn = 2, bipartite_levels = "separate")
  expect_true(is.factor(kn$col1))
  expect_true(is.factor(kn$col2))
  expect_equal(levels(kn$col1), colnames(y))
  expect_equal(levels(kn$col2), colnames(x))
})

test_that("bicor(x, y) dense output is correctly indexed for rectangular inputs", {
  set.seed(707)
  x <- matrix(rnorm(60), nrow = 12, ncol = 5)
  y <- matrix(rnorm(36), nrow = 12, ncol = 3)

  out <- bicor(x, y, pairwise.complete.obs = TRUE, min_overlap = 1)
  expect_identical(dim(out), c(ncol(x), ncol(y)))

  for (i in seq_len(ncol(x))) {
    for (j in seq_len(ncol(y))) {
      ref <- bicor(cbind(x[, i], y[, j]), pairwise.complete.obs = TRUE, min_overlap = 1)[1, 2]
      expect_equal(out[i, j], ref, tolerance = 1e-8)
    }
  }
})

test_that("bicor_knn supports mixed dense and dgCMatrix bipartite inputs", {
  skip_if_not_installed("Matrix")
  set.seed(808)
  x_sparse <- Matrix::rsparsematrix(30, 5, density = 0.45)
  y_dense <- matrix(rnorm(90), nrow = 30, ncol = 3)
  colnames(x_sparse) <- paste0("x", seq_len(ncol(x_sparse)))
  colnames(y_dense) <- paste0("y", seq_len(ncol(y_dense)))

  mixed <- bicor_knn(x_sparse, y_dense, knn = 2, threshold = -Inf,
                     min_overlap = 1, bipartite_levels = "separate")
  ref <- bicor_knn(as.matrix(x_sparse), y_dense, knn = 2, threshold = -Inf,
                   min_overlap = 1, bipartite_levels = "separate")

  normalize_knn <- function(z) {
    z$col1 <- as.integer(z$col1)
    z$col2 <- as.integer(z$col2)
    z[order(z$col1, z$rank, z$col2), , drop = FALSE]
  }

  expect_equal(normalize_knn(mixed), normalize_knn(ref), tolerance = 1e-8)

  y_sparse <- Matrix::rsparsematrix(30, 3, density = 0.5)
  x_dense <- matrix(rnorm(150), nrow = 30, ncol = 5)
  colnames(x_dense) <- paste0("dx", seq_len(ncol(x_dense)))
  colnames(y_sparse) <- paste0("sy", seq_len(ncol(y_sparse)))

  mixed_rev <- bicor_knn(x_dense, y_sparse, knn = 2, threshold = -Inf,
                         min_overlap = 1, bipartite_levels = "separate")
  ref_rev <- bicor_knn(x_dense, as.matrix(y_sparse), knn = 2, threshold = -Inf,
                       min_overlap = 1, bipartite_levels = "separate")

  expect_equal(normalize_knn(mixed_rev), normalize_knn(ref_rev), tolerance = 1e-8)
})
