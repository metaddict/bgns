# Internal reproducible benchmarks; excluded from the source-package tarball.
# Run a case in a fresh process, setting R_LIBS_USER to the build to compare:
# Rscript --vanilla tools/benchmark.R dense-wide results.csv
# Use /usr/bin/time -l on macOS or /usr/bin/time -v on Linux to measure peak RSS.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
suppressPackageStartupMessages(library(bgns))
cases <- list(
  "dense-small" = list(n = 600L, p = 250L, mode = "dense", missing = 0, k = 10L, mem = 16L),
  "dense-wide" = list(n = 600L, p = 1000L, mode = "dense", missing = 0, k = 10L, mem = 16L),
  "dense-tall" = list(n = 2400L, p = 250L, mode = "dense", missing = 0, k = 50L, mem = 64L),
  "sparse" = list(n = 600L, p = 250L, mode = "sparse", missing = 0, k = 10L, mem = 16L),
  "sparse-missing" = list(n = 600L, p = 250L, mode = "sparse", missing = 0.05, k = 10L, mem = 16L),
  "mixed" = list(n = 600L, p = 250L, mode = "mixed", missing = 0.05, k = 10L, mem = 64L)
)
cfg <- cases[[args[1]]]
if (is.null(cfg)) stop("Unknown benchmark case")
Sys.setenv(BGNS_MEM_MB = cfg$mem, BGNS_NUM_THREADS = 1)
set.seed(9241)
x <- matrix(rnorm(cfg$n * cfg$p), cfg$n, cfg$p)
if (cfg$mode != "dense") x[runif(length(x)) < 0.3] <- 0
if (cfg$missing > 0) x[sample.int(length(x), floor(length(x) * cfg$missing))] <- NA_real_
stopifnot(all(apply(x, 2, mad, na.rm = TRUE) > 0))
y <- if (cfg$mode == "mixed") matrix(rnorm(cfg$n * 75), cfg$n, 75) else NULL
input <- if (cfg$mode == "dense") x else Matrix::Matrix(x, sparse = TRUE)
run <- function() bicor_knn(input, y, knn = cfg$k, threshold = 0)
z <- run()
stopifnot(nrow(z) > 0, all(is.finite(z$val)), all(z$rank <= cfg$k))
# Match a small subset to dense output to catch invalid benchmark workloads.
targets <- if (is.null(y)) x[, 1:5, drop = FALSE] else y[, 1:5, drop = FALSE]
reference <- bicor(x, targets)
for (j in 1:5) {
  eligible <- which(is.finite(reference[, j]) & reference[, j] > 0)
  if (is.null(y)) eligible <- setdiff(eligible, j)
  take <- head(eligible[order(-reference[eligible, j], eligible)], cfg$k)
  actual <- z[z$col1 == j, ]
  stopifnot(identical(as.integer(actual$col2), take),
            isTRUE(all.equal(actual$val, unname(reference[take, j]), tolerance = 1e-10)))
}
timings <- replicate(5, unname(system.time(run())[["elapsed"]]))
result <- data.frame(case = args[1], version = as.character(packageVersion("bgns")),
                     n = cfg$n, p = cfg$p, mode = cfg$mode, missing = cfg$missing,
                     k = cfg$k, scratch_mb = cfg$mem, median_seconds = median(timings),
                     min_seconds = min(timings), max_seconds = max(timings), edges = nrow(z))
write.csv(result, args[2], row.names = FALSE)
print(result)
