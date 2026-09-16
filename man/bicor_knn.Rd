\name{bicor_knn}
\alias{bicor_knn}
\title{k-Nearest Neighbors by Biweight Midcorrelation}
\description{
Compute exact top-k neighbors per target column using biweight midcorrelation.
The default streaming algorithm avoids storing a full dense similarity matrix.
}
\usage{
bicor_knn(x, y = NULL, knn, pairwise.complete.obs = TRUE,
          threshold = -Inf, use_intersection_denominator = FALSE,
          direct_sparse = TRUE, bipartite_levels = c("strict", "separate"),
          min_overlap = 3)
}
\arguments{
\item{x}{Numeric matrix or sparse matrix coercible to \code{dgCMatrix} with rows as observations and columns
as source items.}
\item{y}{Optional numeric matrix or sparse matrix coercible to \code{dgCMatrix} with the same number of rows
as \code{x}. When supplied, neighbors are selected from columns of \code{x} for
each target column of \code{y}.}
\item{knn}{Integer. Number of neighbors per target column.}
\item{pairwise.complete.obs}{Logical. If \code{TRUE}, compute each edge on the
finite overlap for that column pair. If \code{FALSE}, only pairs where both
preprocessed columns are complete are eligible; pairs involving incomplete
columns are omitted from the KNN output.}
\item{threshold}{Numeric. Signed cutoff applied before top-k selection. Candidate
bicor values must be strictly greater than \code{threshold}; for example,
\code{threshold = 0} keeps only positive bicor edges.}
\item{use_intersection_denominator}{Logical. Use overlap-specific denominators
for incomplete or sparse paths.}
\item{direct_sparse}{Logical. If \code{TRUE}, stream top-k edges without dense
similarity intermediates. If \code{FALSE}, attempt a full dense correlation step
when the memory budget permits, otherwise fall back to streaming.}
\item{bipartite_levels}{One of \code{"strict"} or \code{"separate"}. Use
\code{"separate"} when \code{x} and \code{y} have distinct column-name levels.}
\item{min_overlap}{Integer. Minimum number of shared finite observations required
between two columns. The requirement also applies when both columns are fully
finite. Defaults to 3.}
}
\details{
Rows are observations and columns are items. With \code{y = NULL}, the result is
a directed KNN graph among columns of \code{x}, with self-neighbors excluded. With
\code{y} supplied, \code{col1} is a target column of \code{y} and \code{col2} is
a selected neighbor column of \code{x}.

The function returns exact top-k neighbors under the implemented bicor score. It
is memory-efficient because it keeps only candidate top-k buffers, but it is not
an approximate subquadratic nearest-neighbor index. Sparse inputs other than \code{dgCMatrix} are coerced to \code{dgCMatrix}. Dense and sparse inputs may
be mixed in bipartite calls, for example a dense matrix \code{x} and sparse
\code{y}, or sparse \code{x} and dense \code{y}.
}
\value{
A data frame with columns \code{col1}, \code{col2}, \code{val}, and
\code{rank}. \code{val} is the bicor score and \code{rank} is the neighbor rank
within each \code{col1} target.
}
\examples{
set.seed(2)
x <- matrix(rnorm(80), nrow = 16, ncol = 5)
x[1, 2] <- NA_real_

kn <- bicor_knn(x, knn = 2, threshold = -Inf)
kn

kn_complete <- bicor_knn(x, knn = 2, pairwise.complete.obs = FALSE)
kn_complete

xs <- Matrix::rsparsematrix(16, 5, density = 0.4)
bicor_knn(xs, knn = 2, threshold = 0)

y <- matrix(rnorm(48), nrow = 16, ncol = 3)
bicor_knn(xs, y, knn = 2, threshold = -Inf)
}
\seealso{\code{\link{bicor}}}
\author{Aditya Kshirsagar}
