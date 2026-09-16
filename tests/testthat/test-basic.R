test_that("sanity - bicor dense returns matrix", {
  set.seed(42)
  x <- matrix(rnorm(2000), nrow = 100, ncol = 20)
  x[sample(length(x), 50)] <- NA_real_
  cm <- bicor(x)
  expect_true(is.matrix(cm))
  expect_identical(dim(cm), c(ncol(x), ncol(x)))
  expect_true(all(is.na(diag(cm)) | diag(cm) == 1 | is.finite(diag(cm))))
})

test_that("bicor validates scalar arguments", {
  x <- matrix(rnorm(50), nrow = 10)
  expect_error(bicor(x, pairwise.complete.obs = NA), "pairwise.complete.obs")
  expect_error(bicor(x, min_overlap = 0), "min_overlap")
  expect_warning(bicor(x, spearman = TRUE), "ignored")
})

test_that("bicor tidy filters by absolute threshold", {
  set.seed(1)
  x <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  td <- bicor(x, tidy = TRUE, threshold = 0.2)
  expect_true(is.data.frame(td))
  expect_identical(names(td), c("col1", "col2", "cor"))
  expect_true(all(abs(td$cor) >= 0.2))
})

test_that("dgCMatrix tidy works", {
  skip_if_not_installed("Matrix")
  set.seed(3)
  xs <- Matrix::rsparsematrix(60, 10, density = 0.2)
  td <- bicor(xs, tidy = TRUE, threshold = 0.0, min_overlap = 1)
  expect_true(is.data.frame(td))
  expect_identical(names(td), c("col1", "col2", "cor"))
})

test_that("bicor_knn basic and validation", {
  set.seed(5)
  x <- matrix(rnorm(1500), nrow = 100, ncol = 15)
  kn <- bicor_knn(x, knn = 3, threshold = 0.0)
  expect_true(is.data.frame(kn))
  expect_identical(names(kn), c("col1", "col2", "val", "rank"))
  expect_true(all(kn$rank %in% 1:3))

  expect_error(bicor_knn(x, knn = 0), "knn")
  expect_error(bicor_knn(x, knn = 2, direct_sparse = NA), "direct_sparse")
})

test_that("bicor_knn signed threshold keeps positive edges", {
  set.seed(6)
  x <- matrix(rnorm(600), nrow = 60, ncol = 10)
  kn <- bicor_knn(x, knn = 3, threshold = 0)
  expect_true(all(kn$val > 0))
})
