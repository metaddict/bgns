# bgns 0.4.2

* Prepared the source tree for CRAN review: corrected package metadata and the source-level CITATION evaluation.
* Updated sparse Matrix coercion to avoid the deprecated direct triplet-to-`dgCMatrix` route.
* Undefined self-correlations for constant, all-missing, or ineligible columns now remain `NA` rather than being forced to 1.
* Positive-integer arguments now reject non-integral values instead of silently truncating them.
* Applied `min_overlap` consistently to fully finite dense fast paths as well as NA-aware paths.
* Fixed the NA-aware sparse self-KNN path, including overlap-specific denominators and stored non-finite values.
* Added internal engine helpers for controlled parallel loops, thread limits, runtime SIMD detection, blocked panel traversal, tiny top-k maintenance, sparse `dgCMatrix` summaries, and safe interrupt checks.
* Renamed the mixed dense/sparse native KNN entry point to `C_bicor_knn_spdem`.
* Kept the public API focused on `bicor()` and `bicor_knn()` while strengthening the native execution layer.
* Preserved exact bicor and exact KNN semantics while improving maintainability and long-running job interruptibility.

# bgns 0.3.5

* Fixed dense `bicor(x, y)` matrix indexing for rectangular `x`/`y` inputs.
* Added native mixed dense/`dgCMatrix` support for `bicor_knn(x, y)`.
* Made dense bipartite KNN factor-level handling match the sparse path.
* Linked BLAS portably through `$(BLAS_LIBS) $(FLIBS)` on Unix-like and Windows builds.
* Updated the optional WGCNA parity test to use pairwise-complete observations and a single WGCNA thread.

# bgns 0.3.3

* Prepared package metadata, documentation, tests, and vignette for repository review.
* Added runnable examples for exported functions.
* Clarified signed-threshold semantics in `bicor_knn()`.
* Clarified KNN edge direction and bipartite behavior.
* Honored `pairwise.complete.obs` in the native KNN interface.
* Removed generated vignette artifacts from the source tree.
* Removed hard-coded local macOS SDK paths from build flags.
* Set an explicit C++ standard and capped default OpenMP usage to at most two threads.
