test_that("undefined self-correlations remain NA", {
  x <- cbind(
    valid = 1:6,
    constant = rep(2, 6),
    incomplete = c(1:5, NA_real_),
    all_missing = rep(NA_real_, 6)
  )

  pair <- bicor(x, pairwise.complete.obs = TRUE, min_overlap = 3)
  expect_equal(pair[1, 1], 1)
  expect_true(is.na(pair[2, 2]))
  expect_equal(pair[3, 3], 1)
  expect_true(is.na(pair[4, 4]))

  complete <- bicor(x, pairwise.complete.obs = FALSE, min_overlap = 3)
  expect_equal(complete[1, 1], 1)
  expect_true(is.na(complete[2, 2]))
  expect_true(is.na(complete[3, 3]))
  expect_true(is.na(complete[4, 4]))
})

test_that("positive integer arguments reject fractional values", {
  x <- matrix(rnorm(50), nrow = 10)
  expect_error(bicor(x, min_overlap = 2.5), "positive integer")
  expect_error(bicor_knn(x, knn = 2.5), "positive integer")
  expect_error(bicor_knn(x, knn = 2, min_overlap = 1.2), "positive integer")
})

test_that("min_overlap applies to fully finite fast paths", {
  x <- cbind(
    a = c(1, 2, 4, 8),
    b = c(3, 1, 7, 5),
    c = c(2, 6, 1, 9)
  )

  expect_true(all(is.na(bicor(x, min_overlap = 5))))
  expect_equal(nrow(bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 5)), 0L)
  expect_equal(nrow(bicor_knn(x, knn = 2, threshold = -Inf, min_overlap = 5)), 0L)
})

test_that("triplet sparse input avoids deprecated coercion", {
  skip_if_not_installed("Matrix")
  x <- Matrix::sparseMatrix(
    i = c(1, 2, 4, 6, 1, 3, 5),
    j = c(1, 1, 1, 1, 2, 2, 2),
    x = c(1, 2, 3, 4, 5, 6, 7),
    dims = c(6, 3),
    repr = "T"
  )

  expect_no_warning(
    out <- bicor(x, tidy = TRUE, threshold = -Inf, min_overlap = 1)
  )
  expect_s3_class(out, "data.frame")
})

test_that("sparse self-KNN NA-aware path matches dense reference", {
  skip_if_not_installed("Matrix")

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
  dense <- matrix(rnorm(120), nrow = 24, ncol = 5)
  dense[abs(dense) < 0.7] <- 0
  colnames(dense) <- paste0("v", seq_len(ncol(dense)))
  sparse <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "CsparseMatrix")

  kd <- bicor_knn(
    dense, knn = 2, threshold = -Inf, min_overlap = 3,
    use_intersection_denominator = TRUE
  )
  ks <- bicor_knn(
    sparse, knn = 2, threshold = -Inf, min_overlap = 3,
    use_intersection_denominator = TRUE
  )
  expect_equal(normalize_knn(ks), normalize_knn(kd), tolerance = 1e-12)

  dense_na <- dense
  dense_na[2, 2] <- NA_real_
  dense_na[7, 4] <- NA_real_
  sparse_na <- methods::as(Matrix::Matrix(dense_na, sparse = TRUE), "CsparseMatrix")

  kd_na <- bicor_knn(
    dense_na, knn = 2, threshold = -Inf, min_overlap = 3,
    pairwise.complete.obs = TRUE,
    use_intersection_denominator = TRUE
  )
  ks_na <- bicor_knn(
    sparse_na, knn = 2, threshold = -Inf, min_overlap = 3,
    pairwise.complete.obs = TRUE,
    use_intersection_denominator = TRUE
  )
  expect_equal(normalize_knn(ks_na), normalize_knn(kd_na), tolerance = 1e-12)
})
