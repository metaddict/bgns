test_that("WGCNA::bicor equivalence on a small matrix", {
  testthat::skip_if_not_installed("WGCNA")
  set.seed(123)
  m <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  m[sample(length(m), 30)] <- NA_real_
  w <- WGCNA::bicor(m, use = "pairwise.complete.obs", nThreads = 1)
  b <- bicor(m, pairwise.complete.obs = TRUE)

  expect_true(
    isTRUE(all.equal(w, b, tolerance = 1e-7)) ||
      isTRUE(all.equal(w, b, tolerance = 1e-6))
  )
})
