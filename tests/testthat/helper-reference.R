# A deliberately direct implementation of the documented estimator, independent
# of native panel construction, sparse indexing and neighbor selection.
reference_bicor <- function(x, y = x, pairwise = TRUE, intersection = FALSE,
                            min_overlap = 3L) {
  midvariate <- function(v) {
    finite <- is.finite(v)
    out <- rep(NA_real_, length(v))
    if (!any(finite)) return(out)
    center <- stats::median(v[finite])
    scale <- stats::mad(v[finite], center = center, constant = 1.4826)
    if (!is.finite(scale) || scale <= 0) return(out)
    d <- v[finite] - center
    u <- d / (9 * scale)
    out[finite] <- d * pmax(0, 1 - u^2)^2 * (abs(u) < 1)
    out
  }
  a <- lapply(seq_len(ncol(x)), function(i) midvariate(x[, i]))
  b <- lapply(seq_len(ncol(y)), function(i) midvariate(y[, i]))
  ans <- matrix(NA_real_, ncol(x), ncol(y))
  for (i in seq_along(a)) for (j in seq_along(b)) {
    overlap <- is.finite(a[[i]]) & is.finite(b[[j]])
    if (sum(overlap) < min_overlap || (!pairwise && !all(overlap))) next
    aa <- if (intersection) a[[i]][overlap] else a[[i]][is.finite(a[[i]])]
    bb <- if (intersection) b[[j]][overlap] else b[[j]][is.finite(b[[j]])]
    denominator <- sqrt(sum(aa^2)) * sqrt(sum(bb^2))
    if (denominator > 0) ans[i, j] <-
      sum(a[[i]][overlap] * b[[j]][overlap]) / denominator
  }
  ans
}

edge_indices <- function(z) {
  z$col1 <- as.integer(z$col1)
  z$col2 <- as.integer(z$col2)
  z <- z[order(z$col1, z$col2), , drop = FALSE]
  rownames(z) <- NULL
  z
}
