test_that("dense, sparse and mixed results follow the documented estimator", {
  set.seed(71021)
  x <- matrix(rnorm(42 * 6), 42, 6)
  y <- matrix(rnorm(42 * 3), 42, 3)
  x[abs(x) < 0.4] <- 0
  y[abs(y) < 0.4] <- 0
  x[c(2, 17, 48)] <- c(NA, Inf, -Inf)
  y[c(5, 24)] <- NA_real_
  x[, 6] <- 0
  xs <- methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix")
  ys <- methods::as(Matrix::Matrix(y, sparse = TRUE), "generalMatrix")
  for (pairwise in c(FALSE, TRUE)) for (inter in c(FALSE, TRUE)) {
    ref <- reference_bicor(x, y, pairwise, inter)
    expect_equal(unname(bicor(x, y, pairwise.complete.obs = pairwise,
                             use_intersection_denominator = inter)), ref,
                 tolerance = 1e-12)
    for (xx in list(x, xs)) for (yy in list(y, ys)) {
      idx <- which(is.finite(ref) & abs(ref) >= 0.1, arr.ind = TRUE)
      expected <- data.frame(col1 = idx[, 1], col2 = idx[, 2], cor = ref[idx])
      actual <- bicor(xx, yy, tidy = TRUE, threshold = 0.1,
                      pairwise.complete.obs = pairwise,
                      use_intersection_denominator = inter)
      expect_equal(edge_indices(actual), edge_indices(expected), tolerance = 1e-12)
      for (direct in c(FALSE, TRUE)) {
        kn <- bicor_knn(xx, yy, knn = 2, threshold = 0,
                        pairwise.complete.obs = pairwise,
                        use_intersection_denominator = inter, direct_sparse = direct)
        for (j in seq_len(ncol(y))) {
          eligible <- which(is.finite(ref[, j]) & ref[, j] > 0)
          take <- head(eligible[order(-ref[eligible, j], eligible)], 2)
          actual_j <- kn[kn$col1 == j, ]
          expect_equal(as.integer(actual_j$col2), take)
          expect_equal(actual_j$val, unname(ref[take, j]), tolerance = 1e-12)
        }
      }
    }
  }
})

test_that("self tidy edges are unique across panel boundaries", {
  old <- Sys.getenv("BGNS_MEM_MB", unset = NA_character_)
  on.exit(if (is.na(old)) Sys.unsetenv("BGNS_MEM_MB") else
    Sys.setenv(BGNS_MEM_MB = old))
  Sys.setenv(BGNS_MEM_MB = 16)
  x <- matrix(rep(seq_len(65536), 32), 65536, 32)
  for (missing in c(FALSE, TRUE)) {
    if (missing) x[1, 1] <- NA_integer_
    for (xx in list(x, methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix"))) {
      z <- bicor(xx, tidy = TRUE, use_intersection_denominator = !missing)
      expect_equal(nrow(z), choose(ncol(x), 2))
      expect_true(all(z$col1 < z$col2))
      expect_equal(anyDuplicated(z[c("col1", "col2")]), 0L)
    }
  }
  Sys.setenv(BGNS_MEM_MB = 32)
  expect_equal(bicor(x, tidy = TRUE), z, tolerance = 1e-12)
})

test_that("oversized neighbor counts are bounded by eligible candidates", {
  x <- matrix(as.double(seq_len(18)), 6, 3)
  xs <- methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix")
  for (xx in list(x, xs)) for (yy in list(NULL, x, xs))
    for (direct in c(FALSE, TRUE)) for (inter in c(FALSE, TRUE)) {
      actual <- bicor_knn(xx, yy, knn = .Machine$integer.max,
                          direct_sparse = direct, use_intersection_denominator = inter)
      expected <- bicor_knn(xx, yy, knn = 3,
                            direct_sparse = direct, use_intersection_denominator = inter)
      expect_equal(actual, expected, tolerance = 1e-12)
    }
  expect_equal(nrow(bicor_knn(x[, 1, drop = FALSE], knn = .Machine$integer.max)), 0L)
  expect_equal(nrow(bicor_knn(x * 0, knn = .Machine$integer.max)), 0L)
})

test_that("rectangular outputs retain names from the corresponding input", {
  x <- cbind(a = 1:6, b = 6:1)
  y <- matrix(1:6, ncol = 1)
  expect_identical(dimnames(bicor(x, y)), list(colnames(x), NULL))
  expect_identical(dimnames(bicor(y, x)), list(NULL, colnames(x)))
  expect_identical(dimnames(bicor(x, unname(x))), list(colnames(x), NULL))
})

test_that("edge identifiers reject duplicate or missing names", {
  x <- cbind(1:6, 6:1, c(1, 3, 5, 2, 4, 6))
  for (nm in list(c("a", "a", "b"), c("a", NA, "b"))) {
    colnames(x) <- nm
    for (xx in list(x, methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix"))) {
      expect_error(bicor(xx, tidy = TRUE), "unique and non-missing")
      expect_error(bicor_knn(xx, knn = 2), "unique and non-missing")
      expect_error(bicor(unname(x), xx, tidy = TRUE), "unique and non-missing")
    }
  }
})

test_that("finite rescaling preserves correlations and neighbor selection", {
  x <- cbind(1:6, 6:1, c(1, 3, 5, 2, 4, 6))
  for (scale in c(1e-300, 1e-100, 1e80, 1e307)) {
    for (inter in c(FALSE, TRUE)) {
      expect_equal(bicor(x * scale, use_intersection_denominator = inter),
                   bicor(x, use_intersection_denominator = inter), tolerance = 1e-12)
      sx <- methods::as(Matrix::Matrix(x * scale, sparse = TRUE), "generalMatrix")
      expect_equal(bicor(sx, tidy = TRUE, use_intersection_denominator = inter),
                   bicor(x, tidy = TRUE, use_intersection_denominator = inter), tolerance = 1e-12)
      expect_equal(bicor_knn(sx, knn = 2, use_intersection_denominator = inter),
                   bicor_knn(x, knn = 2, use_intersection_denominator = inter), tolerance = 1e-12)
    }
  }
  # Opposite signs can overflow subtraction despite every input being finite.
  z <- cbind(c(-1.7, -1.6, -1.5, 1.4, 1.5, 1.6), c(-1.5, 1.4, -1.6, 1.5, -1.7, 1.6))
  expect_equal(bicor(z * 1e308), bicor(z), tolerance = 1e-12)
  # Very large outliers must have zero weight without contaminating finite pairs.
  z <- x; z[1, 1] <- 1e308
  expect_true(all(is.finite(bicor(z))))
})
