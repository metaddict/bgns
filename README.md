# BGNS: Biweight Graph and Network Statistics
The **bgns** package provides robust statistical methods for correlation analysis designed for graph construction, and network clustering. It implements biweight midcorrelation (bicor), a robust alternative for correlation that is more resistant to outliers.


## Package Overview

Robust biweight midcorrelation (bicor) that supports NA-aware pairwise handling.
---

## Key Capabilities

### Robust Correlation Analysis
- **Biweight midcorrelation (bicor):** Uses median-based robust statistics instead of mean-based methods  
- **NA-aware handling:** Efficient missing data handling with pairwise complete observations  
- **High performance:** C++ core with optimized BLAS paths and optional OpenMP support  

### Graph and Network Analysis
- **KNN graph construction:** Builds K-nearest neighbor graphs from bicor similarities  
- **Graph clustering:** Useful for bootstrap resampling for robust clustering results and clustering algorithms on similarity graphs.

---

## Technical Features
- Compatible with tgstat graph utilities for extended analysis  
- Supports both dense matrices and sparse `Matrix` formats  
- Flexible output formats (matrix or tidy `data.frame`)  
- Memory-efficient algorithms for large datasets  
- Multiple normalization options including intersection-based denominators  

---

## Use Cases
This package is particularly valuable for:
- Genomics and bioinformatics (gene expression analysis)  
- Network biology (co-expression networks)  
- Social network analysis with noisy data  
- Any domain requiring robust correlation analysis with subsequent graph/network analysis  

---

## Installation

The development version of **bgns** from GitHub:

```r
# install.packages("devtools")
devtools::install_github("metaddict/bgns")
