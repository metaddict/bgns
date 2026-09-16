\name{bicor}
\alias{bicor}
\title{Robust Biweight Midcorrelation}
\description{
Compute biweight midcorrelations between columns of \code{x} and, optionally,
between columns of \code{x} and \code{y}. Tidy output streams correlation edges
without materializing a full dense correlation matrix.
}
\usage{
bicor(x, y = NULL, pairwise.complete.obs = TRUE, spearman = FALSE,
      tidy = FALSE, threshold = -Inf, use_intersection_denominator = FALSE,
      min_overlap = 3)
}
\arguments{
\item{x}{Numeric matrix with rows as observations and columns as items. Sparse objects inheriting from \code{sparseMatrix} are coerced to \code{dgCMatrix}; sparse inputs are supported when \code{tidy = TRUE}.}
\item{y}{Optional numeric matrix or sparse matrix coercible to \code{dgCMatrix} with the same number of rows
as \code{x}.}
\item{pairwise.complete.obs}{Logical. If \code{TRUE}, compute each correlation on
the finite pairwise overlap. If \code{FALSE}, pairs involving incomplete columns
are omitted in tidy mode or returned as \code{NA} in dense matrix mode.}
\item{spearman}{Logical. Accepted for API compatibility and ignored; this
function computes biweight midcorrelation.}
\item{tidy}{Logical. If \code{TRUE}, return a data frame with columns
\code{col1}, \code{col2}, and \code{cor}.}
\item{threshold}{Numeric. In tidy mode, keep pairs with
\eqn{|cor| \ge threshold}. The default keeps all finite pairs.}
\item{use_intersection_denominator}{Logical. For sparse or incomplete columns,
use the overlap-specific denominator rather than full-column sums of squares.}
\item{min_overlap}{Integer. Minimum number of shared finite observations required
between two columns. The requirement also applies when both columns are fully
finite. Defaults to 3.}
}
\details{
Column medians and scaled MADs (constant 1.4826) are estimated from finite
observations. Biweight deviations use a tuning multiplier of 9. These estimates
and weights are fixed before pairwise comparisons. The numerator uses shared
finite observations; by default, the denominator uses full-column sums of
squared weighted deviations. Setting \code{use_intersection_denominator = TRUE}
restricts the sums to shared observations without refitting the weights.
Implicit sparse entries are observed zeros. Row matching is positional.

Named edge outputs require unique, non-missing column names within each input.
Unnamed inputs use column indices.

Input orientation is rows as observations and columns as variables, genes, cells,
metacells, or other items to be compared.

When \code{tidy = FALSE}, both inputs must be dense base matrices. Sparse
inputs are coerced to \code{dgCMatrix} and require \code{tidy = TRUE}. The tidy output stores pairwise correlations as
an edge table and is intended for downstream graph construction.

For self-correlation matrices, a diagonal entry is 1 only when that column has
a defined robust variance and satisfies \code{min_overlap}. Constant,
all-missing, and otherwise ineligible columns have \code{NA} on the diagonal.
}
\value{
A dense correlation matrix when \code{tidy = FALSE} and inputs are dense, or a
data frame with columns \code{col1}, \code{col2}, and \code{cor} when
\code{tidy = TRUE}.
}
\examples{
set.seed(1)
x <- matrix(rnorm(60), nrow = 12, ncol = 5)
x[1, 2] <- NA_real_

cm <- bicor(x)
cm

td <- bicor(x, tidy = TRUE, threshold = 0.1)
head(td)

x_sparse <- x
x_sparse[sample.int(length(x_sparse), 12)] <- 0
xs <- Matrix::Matrix(x_sparse, sparse = TRUE)
head(bicor(xs, tidy = TRUE, threshold = 0))
}
\seealso{\code{\link{bicor_knn}}}
\author{Aditya Kshirsagar}
