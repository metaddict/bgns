// src/bicor.cpp
#define USE_FC_LEN_T
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <utility>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <R.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#ifdef length
#undef length
#endif
#ifdef error
#undef error
#endif
#ifndef FCONE
#define FCONE
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdint>

#if defined(__has_include)
#if __has_include("tgstat.h")
#define BGNS_HAVE_TGSTAT 1
#include "tgstat.h"
#include "ProgressReporter.h"
#else
#define BGNS_HAVE_TGSTAT 0
#endif
#else
// conservative default if compiler lacks __has_include
#define BGNS_HAVE_TGSTAT 0
#endif

#if !BGNS_HAVE_TGSTAT
// ------- minimal shims so we compile & run without tgstat headers -------
struct TGStat
{
    TGStat(SEXP) {}
    static TGStat &instance()
    {
        static TGStat t(R_NilValue);
        return t;
    }
};
struct ProgressReporter
{
    ProgressReporter(TGStat &, std::uint64_t) {}
    void next() {}
};
// map tgstat-style helpers to base R API
#ifndef verror
#define verror(...) Rf_error(__VA_ARGS__)
#endif
#ifndef rerror
#define rerror(...) Rf_error(__VA_ARGS__)
#endif
// To mimic tgstat's automatic UNPROTECT-on-return behavior, we track
// the protection count in a function-local variable named __prot_count.
// Each C entrypoint must declare: int __prot_count = 0; near the top.
#ifndef rprotect
#define rprotect(x) do { PROTECT(x); ++__prot_count; } while (0)
#endif
#ifndef rreturn
#define rreturn(x) do { UNPROTECT(__prot_count); return (x); } while (0)
#endif
#ifndef RSaneAllocVector
#define RSaneAllocVector(SEXPTYPE_, N_) Rf_allocVector(SEXPTYPE_, (R_xlen_t)(N_))
#endif
#endif // BGNS_HAVE_TGSTAT

using std::max;
using std::min;
using std::pair;
using std::vector;

namespace
{

    // ---------- robust helpers ----------
    inline double median_inplace(vector<double> &v)
    {
        const size_t n = v.size();
        if (n == 0)
            return NA_REAL;
        const size_t mid = n / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        double m = v[mid];
        if ((n & 1U) == 0U)
        {
            auto it = std::max_element(v.begin(), v.begin() + mid);
            m = 0.5 * (m + *it);
        }
        return m;
    }

    inline double mad_scaled(const vector<double> &vals, double med)
    {
        if (vals.empty())
            return NA_REAL;
        vector<double> tmp(vals.size());
        for (size_t i = 0; i < vals.size(); ++i)
            tmp[i] = std::fabs(vals[i] - med);
        double unscaled = median_inplace(tmp);
        if (!R_FINITE(unscaled) || unscaled <= 0.0)
            return NA_REAL;
        return 1.4826 * unscaled;
    }

    // strict pairwise mid-variates on finite rows only (K = 9)
    struct MidCol
    {
        vector<int> idx;
        vector<double> mv;
        double sumsq;
        bool ok;
    };

    inline MidCol build_midcol_strict(const double *col, int n)
    {
        MidCol out;
        out.idx.reserve(n);
        vector<double> finite;
        finite.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            const double v = col[i];
            if (R_FINITE(v))
            {
                out.idx.push_back(i);
                finite.push_back(v);
            }
        }
        if (finite.empty())
        {
            out.sumsq = 0.0;
            out.ok = false;
            return out;
        }

        vector<double> tmp = finite;
        const double med = median_inplace(tmp);
        const double mad = mad_scaled(finite, med);
        if (!(mad > 0.0))
        {
            out.sumsq = 0.0;
            out.ok = false;
            return out;
        }

        const double inv_k = 1.0 / (9.0 * mad); // tuning K = 9
        out.mv.resize(finite.size());
        double ss = 0.0;
        for (size_t k = 0; k < finite.size(); ++k)
        {
            const double v = finite[k];
            const double u = (v - med) * inv_k;
            const double w = (std::fabs(u) >= 1.0) ? 0.0 : (1.0 - u * u) * (1.0 - u * u);
            const double mv = (v - med) * w;
            out.mv[k] = mv;
            ss += mv * mv;
        }
        out.sumsq = ss;
        out.ok = (ss > 0.0);
        return out;
    }

    // dense (NA-free) mid-variates for BLAS path
    struct DenseCol
    {
        vector<double> mv;
        double sumsq;
        bool ok;
    };

    inline DenseCol build_midcol_dense(const double *col, int n)
    {
        DenseCol out;
        out.mv.resize(n);
        vector<double> vals(n);
        for (int i = 0; i < n; ++i)
        {
            const double v = col[i];
            if (!R_FINITE(v))
            {
                out.mv.clear();
                out.sumsq = 0.0;
                out.ok = false;
                return out;
            }
            vals[i] = v;
        }
        vector<double> tmp = vals;
        const double med = median_inplace(tmp);
        const double mad = mad_scaled(vals, med);
        if (!(mad > 0.0))
        {
            out.mv.clear();
            out.sumsq = 0.0;
            out.ok = false;
            return out;
        }
        const double inv_k = 1.0 / (9.0 * mad);
        double ss = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double v = vals[i];
            const double u = (v - med) * inv_k;
            const double w = (std::fabs(u) >= 1.0) ? 0.0 : (1.0 - u * u) * (1.0 - u * u);
            const double mv = (v - med) * w;
            out.mv[i] = mv;
            ss += mv * mv;
        }
        out.sumsq = ss;
        out.ok = (ss > 0.0);
        return out;
    }

    struct DotStat
    {
        double num, sA, sB;
        int overlap;
    };

    inline DotStat dot_intersect_stats(const MidCol &A, const MidCol &B, bool want_inter)
    {
        size_t i = 0, j = 0;
        double num = 0.0, sa = 0.0, sb = 0.0;
        int ov = 0;
        while (i < A.idx.size() && j < B.idx.size())
        {
            if (A.idx[i] < B.idx[j])
            {
                ++i;
            }
            else if (B.idx[j] < A.idx[i])
            {
                ++j;
            }
            else
            {
                const double va = A.mv[i], vb = B.mv[j];
                num += va * vb;
                if (want_inter)
                {
                    sa += va * va;
                    sb += vb * vb;
                }
                ++ov;
                ++i;
                ++j;
            }
        }
        return {num, sa, sb, ov};
    }

    inline void dgemm_AB(char transA, char transB,
                         const int M, const int N, const int K,
                         const double alpha,
                         const double *A, const int lda,
                         const double *B, const int ldb,
                         const double beta,
                         double *C, const int ldc)
    {
        F77_CALL(dgemm)(&transA, &transB, &M, &N, &K,
                        &alpha, A, &lda, B, &ldb, &beta, C, &ldc FCONE FCONE);
    }

    inline int choose_block_cols(size_t p_left, size_t p_all, size_t n, size_t cap_mb)
    {
        const size_t bytes_cap = cap_mb * (size_t)1024 * (size_t)1024;
        const size_t max_blk = bytes_cap / (sizeof(double) * p_all);
        if (max_blk == 0)
            return 1;
        return (int)std::max<size_t>(1, std::min<size_t>(p_left, max_blk));
    }

} // namespace

extern "C"
{

    // ============================== bicor ==============================
    // bicor(x, y = NULL, pairwise_complete_obs, use_inter_den, envir)
    SEXP C_bicor(SEXP _x, SEXP _y, SEXP _pairwise, SEXP _use_inter_den, SEXP _envir)
    {
        SEXP answer = R_NilValue;
    int __prot_count = 0; // used by rprotect/rreturn shim
        try
        {
            TGStat tgstat(_envir);
            const bool haveY = !Rf_isNull(_y);

            if ((!Rf_isReal(_x) && !Rf_isInteger(_x)) || (haveY && !Rf_isReal(_y) && !Rf_isInteger(_y)))
                verror("\"x\"/\"y\" must be numeric matrices");
            if (!Rf_isMatrix(_x) || (haveY && !Rf_isMatrix(_y)))
                verror("\"x\"/\"y\" must be matrices");

            const int n = Rf_nrows(_x);
            const int px = Rf_ncols(_x);
            const int py = haveY ? Rf_ncols(_y) : px;

            const bool pairwise_complete = Rf_asLogical(_pairwise);
            const bool use_inter_den = Rf_asLogical(_use_inter_den);
            const int min_overlap = 3;

            // Quick NA scan for BLAS
            bool na_free_x = true, na_free_y = true;
            if (Rf_isReal(_x))
            {
                const double *xp = REAL(_x);
                for (int j = 0; j < px && na_free_x; ++j)
                    for (int i = 0; i < n; ++i)
                        if (!R_FINITE(xp[(size_t)j * n + i]))
                        {
                            na_free_x = false;
                            break;
                        }
            }
            else
            {
                const int *xi = INTEGER(_x);
                for (int j = 0; j < px && na_free_x; ++j)
                    for (int i = 0; i < n; ++i)
                        if (xi[(size_t)j * n + i] == NA_INTEGER)
                        {
                            na_free_x = false;
                            break;
                        }
            }
            if (haveY)
            {
                if (Rf_isReal(_y))
                {
                    const double *yp = REAL(_y);
                    for (int j = 0; j < py && na_free_y; ++j)
                        for (int i = 0; i < n; ++i)
                            if (!R_FINITE(yp[(size_t)j * n + i]))
                            {
                                na_free_y = false;
                                break;
                            }
                }
                else
                {
                    const int *yi = INTEGER(_y);
                    for (int j = 0; j < py && na_free_y; ++j)
                        for (int i = 0; i < n; ++i)
                            if (yi[(size_t)j * n + i] == NA_INTEGER)
                            {
                                na_free_y = false;
                                break;
                            }
                }
            }
            const bool na_free = na_free_x && (!haveY || na_free_y);

            // Output
            rprotect(answer = RSaneAllocVector(REALSXP, (uint64_t)px * (uint64_t)py));
            double *ans = REAL(answer);
            for (uint64_t t = 0, T = (uint64_t)px * (uint64_t)py; t < T; ++t)
                ans[t] = NA_REAL;

            ProgressReporter progress(TGStat::instance(), haveY ? (uint64_t)py : (uint64_t)px);

            // BLAS fast path when NA-free and denom by columns (intersection==columns)
            const bool use_blas = na_free && !use_inter_den;

            if (use_blas)
            {
                vector<double> A((size_t)n * px), B;
                vector<double> ssX(px), ssY;

                for (int j = 0; j < px; ++j)
                {
                    vector<double> col(n);
                    if (Rf_isReal(_x))
                    {
                        const double *xp = REAL(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            col[i] = xp[i];
                    }
                    else
                    {
                        const int *xi = INTEGER(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            col[i] = (double)xi[i];
                    }
                    DenseCol dc = build_midcol_dense(col.data(), n);
                    if (!dc.ok)
                    {
                        ssX[j] = 0.0;
                        continue;
                    }
                    ssX[j] = dc.sumsq;
                    for (int i = 0; i < n; ++i)
                        A[(size_t)j * n + i] = dc.mv[i];
                }
                if (haveY)
                {
                    B.resize((size_t)n * py);
                    ssY.resize(py);
                    for (int j = 0; j < py; ++j)
                    {
                        vector<double> col(n);
                        if (Rf_isReal(_y))
                        {
                            const double *yp = REAL(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                col[i] = yp[i];
                        }
                        else
                        {
                            const int *yi = INTEGER(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                col[i] = (double)yi[i];
                        }
                        DenseCol dc = build_midcol_dense(col.data(), n);
                        if (!dc.ok)
                        {
                            ssY[j] = 0.0;
                            continue;
                        }
                        ssY[j] = dc.sumsq;
                        for (int i = 0; i < n; ++i)
                            B[(size_t)j * n + i] = dc.mv[i];
                    }
                }
                else
                {
                    B = A;
                    ssY = ssX;
                }

                const size_t cap_mb = 64;
                for (int j0 = 0; j0 < py;)
                {
                    const int blk = choose_block_cols((size_t)(py - j0), (size_t)px, (size_t)n, cap_mb);
                    vector<double> C((size_t)px * blk, 0.0);
                    dgemm_AB('T', 'N', px, blk, n, 1.0, A.data(), n, B.data() + (size_t)j0 * n, n, 0.0, C.data(), px);

                    for (int jb = 0; jb < blk; ++jb)
                    {
                        const int j = j0 + jb;
                        const double ss_t = ssY[j];
                        if (!(ss_t > 0.0))
                            continue;
                        for (int i = 0; i < px; ++i)
                        {
                            const double ss_s = ssX[i];
                            if (!(ss_s > 0.0))
                            {
                                ans[(size_t)i * py + j] = NA_REAL;
                                continue;
                            }
                            const double num = C[(size_t)jb * px + i];
                            const double den = std::sqrt(ss_s * ss_t);
                            if (den > 0.0)
                                ans[(size_t)i * py + j] = num / den;
                        }
                    }
                    j0 += blk;
                    progress.next();
                }

                if (!haveY)
                {
                    for (int j = 0; j < px; ++j)
                        ans[(size_t)j * px + j] = (n > 1 ? 1.0 : NA_REAL);
                }
            }
            else
            {
                // STRICT or ALL.OBS path
                vector<MidCol> MX(px), MY;
                vector<char> fullX(px, 1), fullY;
                for (int j = 0; j < px; ++j)
                {
                    if (Rf_isReal(_x))
                    {
                        MX[j] = build_midcol_strict(REAL(_x) + (size_t)j * n, n);
                    }
                    else
                    {
                        vector<double> tmp(n);
                        const int *xi = INTEGER(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            tmp[i] = (xi[i] == NA_INTEGER) ? NA_REAL : (double)xi[i];
                        MX[j] = build_midcol_strict(tmp.data(), n);
                    }
                    fullX[j] = (MX[j].idx.size() == (size_t)n) ? 1 : 0;
                }
                if (haveY)
                {
                    MY.resize(py);
                    fullY.assign(py, 1);
                    for (int j = 0; j < py; ++j)
                    {
                        if (Rf_isReal(_y))
                        {
                            MY[j] = build_midcol_strict(REAL(_y) + (size_t)j * n, n);
                        }
                        else
                        {
                            vector<double> tmp(n);
                            const int *yi = INTEGER(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                tmp[i] = (yi[i] == NA_INTEGER) ? NA_REAL : (double)yi[i];
                            MY[j] = build_midcol_strict(tmp.data(), n);
                        }
                        fullY[j] = (MY[j].idx.size() == (size_t)n) ? 1 : 0;
                    }
                }

                if (!haveY)
                {
                    for (int j = 0; j < px; ++j)
                        ans[(size_t)j * px + j] = (n > 1 ? 1.0 : NA_REAL);
                    for (int j = 0; j < px; ++j)
                    {
                        const MidCol &A = MX[j];
                        for (int k = j + 1; k < px; ++k)
                        {
                            const MidCol &B = MX[k];
                            double r = NA_REAL;

                            if (pairwise_complete)
                            {
                                if (A.ok && B.ok)
                                {
                                    auto st = dot_intersect_stats(A, B, use_inter_den);
                                    if (st.overlap >= min_overlap)
                                    {
                                        const double den = use_inter_den ? std::sqrt(st.sA * st.sB)
                                                                         : std::sqrt(A.sumsq * B.sumsq);
                                        if (den > 0.0)
                                            r = st.num / den;
                                    }
                                }
                            }
                            else
                            {
                                // all.obs: only compute if both columns have no missing (full length)
                                if (A.ok && B.ok && fullX[j] && fullX[k])
                                {
                                    const double den = std::sqrt(A.sumsq * B.sumsq);
                                    if (den > 0.0)
                                    {
                                        auto st = dot_intersect_stats(A, B, false);
                                        r = st.num / den;
                                    }
                                }
                            }

                            ans[(size_t)j * px + k] = r;
                            ans[(size_t)k * px + j] = r;
                        }
                        progress.next();
                    }
                }
                else
                {
                    for (int j = 0; j < py; ++j)
                    {
                        const MidCol &T = MY[j];
                        for (int i = 0; i < px; ++i)
                        {
                            const MidCol &S = MX[i];
                            double r = NA_REAL;

                            if (pairwise_complete)
                            {
                                if (S.ok && T.ok)
                                {
                                    auto st = dot_intersect_stats(S, T, use_inter_den);
                                    if (st.overlap >= min_overlap)
                                    {
                                        const double den = use_inter_den ? std::sqrt(st.sA * st.sB)
                                                                         : std::sqrt(S.sumsq * T.sumsq);
                                        if (den > 0.0)
                                            r = st.num / den;
                                    }
                                }
                            }
                            else
                            {
                                // all.obs: only when both columns are full length
                                if (S.ok && T.ok && fullX[i] && fullY[j])
                                {
                                    const double den = std::sqrt(S.sumsq * T.sumsq);
                                    if (den > 0.0)
                                    {
                                        auto st = dot_intersect_stats(S, T, false);
                                        r = st.num / den;
                                    }
                                }
                            }

                            ans[(size_t)i * py + j] = r;
                        }
                        progress.next();
                    }
                }
            }

            // dim / dimnames
            SEXP rdim;
            rprotect(rdim = RSaneAllocVector(INTSXP, 2));
            INTEGER(rdim)[0] = px;
            INTEGER(rdim)[1] = py;
            Rf_setAttrib(answer, R_DimSymbol, rdim);

            // Attach dimnames only if we actually have column names on x or y
            SEXP xdn = Rf_getAttrib(_x, R_DimNamesSymbol);
            SEXP xcols = (Rf_isNull(xdn) || Rf_xlength(xdn) != 2) ? R_NilValue : VECTOR_ELT(xdn, 1);
            SEXP ycols = R_NilValue;
            if (haveY)
            {
                SEXP ydn = Rf_getAttrib(_y, R_DimNamesSymbol);
                if (!Rf_isNull(ydn) && Rf_xlength(ydn) == 2)
                    ycols = VECTOR_ELT(ydn, 1);
            }
            const bool have_xcols = !Rf_isNull(xcols);
            const bool have_ycols = !Rf_isNull(ycols);
            if (have_xcols || (haveY && have_ycols))
            {
                SEXP rdimnames;
                rprotect(rdimnames = RSaneAllocVector(VECSXP, 2));
                SET_VECTOR_ELT(rdimnames, 0, have_xcols ? xcols : R_NilValue);
                if (haveY)
                {
                    // Prefer y colnames if provided; else reuse x colnames
                    SET_VECTOR_ELT(rdimnames, 1, have_ycols ? ycols : (have_xcols ? xcols : R_NilValue));
                }
                else
                {
                    SET_VECTOR_ELT(rdimnames, 1, have_xcols ? xcols : R_NilValue);
                }
                Rf_setAttrib(answer, R_DimNamesSymbol, rdimnames);
            }
            else if (haveY)
            {
                // For cross-matrix mode with no names on either side, mirror tgstat by
                // attaching an explicit dimnames list of two NULLs.
                SEXP rdimnames;
                rprotect(rdimnames = RSaneAllocVector(VECSXP, 2));
                SET_VECTOR_ELT(rdimnames, 0, R_NilValue);
                SET_VECTOR_ELT(rdimnames, 1, R_NilValue);
                Rf_setAttrib(answer, R_DimNamesSymbol, rdimnames);
            }
        }
        catch (const std::bad_alloc &)
        {
            rerror("Out of memory");
        }
#if BGNS_HAVE_TGSTAT
        catch (TGLException &e)
        {
            rerror("%s", e.msg());
#else
        catch (const std::exception &e)
        {
            rerror("%s", e.what());
#endif
        }
        rreturn(answer);
    }

    // ============================== bicor_knn ==============================
    // bicor_knn(x, y=NULL, knn, threshold, use_inter_den, envir)
    SEXP C_bicor_knn(SEXP _x, SEXP _y, SEXP _knn, SEXP _threshold, SEXP _use_inter_den, SEXP _envir)
    {
        SEXP ans_df = R_NilValue;
    int __prot_count = 0; // used by rprotect/rreturn shim
        try
        {
            TGStat tgstat(_envir);
            const bool haveY = !Rf_isNull(_y);

            if ((!Rf_isReal(_x) && !Rf_isInteger(_x)) || (haveY && !Rf_isReal(_y) && !Rf_isInteger(_y)))
                verror("\"x\"/\"y\" must be numeric matrices");
            if (!Rf_isMatrix(_x) || (haveY && !Rf_isMatrix(_y)))
                verror("\"x\"/\"y\" must be matrices");

            const int n = Rf_nrows(_x);
            const int px = Rf_ncols(_x);
            const int py = haveY ? Rf_ncols(_y) : px;

            if (Rf_xlength(_knn) != 1)
                verror("\"knn\" must be length 1");
            const int K = Rf_asInteger(_knn);
            if (K < 1)
                verror("\"knn\" must be >= 1");
            const double threshold = Rf_asReal(_threshold);
            const bool use_inter_den = Rf_asLogical(_use_inter_den);
            const int min_overlap = 3;

            // Quick NA scan
            bool na_free_x = true, na_free_y = true;
            if (Rf_isReal(_x))
            {
                const double *xp = REAL(_x);
                for (int j = 0; j < px && na_free_x; ++j)
                    for (int i = 0; i < n; ++i)
                        if (!R_FINITE(xp[(size_t)j * n + i]))
                        {
                            na_free_x = false;
                            break;
                        }
            }
            else
            {
                const int *xi = INTEGER(_x);
                for (int j = 0; j < px && na_free_x; ++j)
                    for (int i = 0; i < n; ++i)
                        if (xi[(size_t)j * n + i] == NA_INTEGER)
                        {
                            na_free_x = false;
                            break;
                        }
            }
            if (haveY)
            {
                if (Rf_isReal(_y))
                {
                    const double *yp = REAL(_y);
                    for (int j = 0; j < py && na_free_y; ++j)
                        for (int i = 0; i < n; ++i)
                            if (!R_FINITE(yp[(size_t)j * n + i]))
                            {
                                na_free_y = false;
                                break;
                            }
                }
                else
                {
                    const int *yi = INTEGER(_y);
                    for (int j = 0; j < py && na_free_y; ++j)
                        for (int i = 0; i < n; ++i)
                            if (yi[(size_t)j * n + i] == NA_INTEGER)
                            {
                                na_free_y = false;
                                break;
                            }
                }
            }
            const bool use_blas = na_free_x && (!haveY || na_free_y) && !use_inter_den;

            ProgressReporter progress(TGStat::instance(), (uint64_t)(haveY ? py : px));

            vector<int> col1;
            col1.reserve((size_t)(haveY ? py : px) * min(K, px));
            vector<int> col2;
            col2.reserve(col1.capacity());
            vector<int> rankv;
            rankv.reserve(col1.capacity());
            vector<double> val;
            val.reserve(col1.capacity());

            if (use_blas)
            {
                // BLAS route (NA-free)
                vector<double> A((size_t)n * px), B;
                vector<double> ssX(px), ssY;

                for (int j = 0; j < px; ++j)
                {
                    vector<double> col(n);
                    if (Rf_isReal(_x))
                    {
                        const double *xp = REAL(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            col[i] = xp[i];
                    }
                    else
                    {
                        const int *xi = INTEGER(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            col[i] = (double)xi[i];
                    }
                    DenseCol dc = build_midcol_dense(col.data(), n);
                    ssX[j] = dc.ok ? dc.sumsq : 0.0;
                    if (dc.ok)
                        for (int i = 0; i < n; ++i)
                            A[(size_t)j * n + i] = dc.mv[i];
                }
                if (haveY)
                {
                    B.resize((size_t)n * py);
                    ssY.resize(py);
                    for (int j = 0; j < py; ++j)
                    {
                        vector<double> col(n);
                        if (Rf_isReal(_y))
                        {
                            const double *yp = REAL(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                col[i] = yp[i];
                        }
                        else
                        {
                            const int *yi = INTEGER(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                col[i] = (double)yi[i];
                        }
                        DenseCol dc = build_midcol_dense(col.data(), n);
                        ssY[j] = dc.ok ? dc.sumsq : 0.0;
                        if (dc.ok)
                            for (int i = 0; i < n; ++i)
                                B[(size_t)j * n + i] = dc.mv[i];
                    }
                }
                else
                {
                    B = A;
                    ssY = ssX;
                }

                const size_t cap_mb = 64;
                for (int j0 = 0; j0 < py;)
                {
                    const int blk = choose_block_cols((size_t)(py - j0), (size_t)px, (size_t)n, cap_mb);
                    vector<double> C((size_t)px * blk, 0.0);
                    dgemm_AB('T', 'N', px, blk, n, 1.0, A.data(), n, B.data() + (size_t)j0 * n, n, 0.0, C.data(), px);

                    for (int jb = 0; jb < blk; ++jb)
                    {
                        const int tgt = j0 + jb;
                        const double ss_t = ssY[tgt];
                        if (!(ss_t > 0.0))
                        {
                            continue;
                        }
                        vector<pair<int, double>> buf;
                        buf.reserve(px);
                        for (int src = 0; src < px; ++src)
                        {
                            if (!haveY && src == tgt)
                                continue;
                            const double ss_s = ssX[src];
                            if (!(ss_s > 0.0))
                                continue;
                            const double num = C[(size_t)jb * px + src];
                            const double den = std::sqrt(ss_s * ss_t);
                            if (den <= 0.0)
                                continue;
                            const double r = num / den;
                            if (R_FINITE(r) && r > threshold)
                                buf.emplace_back(src, r);
                        }
                        if (!buf.empty())
                        {
                            int keep = (int)min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin() + keep - 1, buf.end(),
                                             [](const pair<int, double> &a, const pair<int, double> &b)
                                             { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            buf.resize(keep);
                            std::sort(buf.begin(), buf.end(),
                                      [](const pair<int, double> &a, const pair<int, double> &b)
                                      { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            for (int i = 0; i < keep; ++i)
                            {
                                col1.push_back(tgt + 1);
                                col2.push_back(buf[i].first + 1);
                                val.push_back(buf[i].second);
                                rankv.push_back(i + 1);
                            }
                        }
                    }
                    j0 += blk;
                    progress.next();
                }
            }
            else
            {
                // strict pairwise path
                vector<MidCol> MX(px), MY;
                for (int j = 0; j < px; ++j)
                {
                    if (Rf_isReal(_x))
                        MX[j] = build_midcol_strict(REAL(_x) + (size_t)j * n, n);
                    else
                    {
                        vector<double> tmp(n);
                        const int *xi = INTEGER(_x) + (size_t)j * n;
                        for (int i = 0; i < n; ++i)
                            tmp[i] = (xi[i] == NA_INTEGER) ? NA_REAL : (double)xi[i];
                        MX[j] = build_midcol_strict(tmp.data(), n);
                    }
                }
                if (haveY)
                {
                    MY.resize(py);
                    for (int j = 0; j < py; ++j)
                    {
                        if (Rf_isReal(_y))
                            MY[j] = build_midcol_strict(REAL(_y) + (size_t)j * n, n);
                        else
                        {
                            vector<double> tmp(n);
                            const int *yi = INTEGER(_y) + (size_t)j * n;
                            for (int i = 0; i < n; ++i)
                                tmp[i] = (yi[i] == NA_INTEGER) ? NA_REAL : (double)yi[i];
                            MY[j] = build_midcol_strict(tmp.data(), n);
                        }
                    }
                }

                if (!haveY)
                {
                    for (int tgt = 0; tgt < px; ++tgt)
                    {
                        const MidCol &T = MX[tgt];
                        if (!T.ok || (int)T.idx.size() < min_overlap)
                        {
                            progress.next();
                            continue;
                        }
                        vector<pair<int, double>> buf;
                        buf.reserve(px - 1);
                        for (int src = 0; src < px; ++src)
                        {
                            if (src == tgt)
                                continue;
                            const MidCol &S = MX[src];
                            if (!S.ok || (int)S.idx.size() < min_overlap)
                                continue;
                            auto st = dot_intersect_stats(S, T, use_inter_den);
                            if (st.overlap < min_overlap)
                                continue;
                            const double den = use_inter_den ? std::sqrt(st.sA * st.sB) : std::sqrt(S.sumsq * T.sumsq);
                            if (den <= 0.0)
                                continue;
                            const double r = st.num / den;
                            if (R_FINITE(r) && r > threshold)
                                buf.emplace_back(src, r);
                        }
                        if (!buf.empty())
                        {
                            int keep = (int)min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin() + keep - 1, buf.end(),
                                             [](const pair<int, double> &a, const pair<int, double> &b)
                                             { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            buf.resize(keep);
                            std::sort(buf.begin(), buf.end(),
                                      [](const pair<int, double> &a, const pair<int, double> &b)
                                      { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            for (int i = 0; i < keep; ++i)
                            {
                                col1.push_back(tgt + 1);
                                col2.push_back(buf[i].first + 1);
                                val.push_back(buf[i].second);
                                rankv.push_back(i + 1);
                            }
                        }
                        progress.next();
                    }
                }
                else
                {
                    for (int tgt = 0; tgt < py; ++tgt)
                    {
                        const MidCol &T = MY[tgt];
                        if (!T.ok || (int)T.idx.size() < min_overlap)
                        {
                            progress.next();
                            continue;
                        }
                        vector<pair<int, double>> buf;
                        buf.reserve(px);
                        for (int src = 0; src < px; ++src)
                        {
                            const MidCol &S = MX[src];
                            if (!S.ok || (int)S.idx.size() < min_overlap)
                                continue;
                            auto st = dot_intersect_stats(S, T, use_inter_den);
                            const double den = use_inter_den ? std::sqrt(st.sA * st.sB) : std::sqrt(S.sumsq * T.sumsq);
                            if (st.overlap < min_overlap || den <= 0.0)
                                continue;
                            const double r = st.num / den;
                            if (R_FINITE(r) && r > threshold)
                                buf.emplace_back(src, r);
                        }
                        if (!buf.empty())
                        {
                            int keep = (int)min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin() + keep - 1, buf.end(),
                                             [](const pair<int, double> &a, const pair<int, double> &b)
                                             { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            buf.resize(keep);
                            std::sort(buf.begin(), buf.end(),
                                      [](const pair<int, double> &a, const pair<int, double> &b)
                                      { return a.second > b.second || (a.second == b.second && a.first < b.first); });
                            for (int i = 0; i < keep; ++i)
                            {
                                col1.push_back(tgt + 1);
                                col2.push_back(buf[i].first + 1);
                                val.push_back(buf[i].second);
                                rankv.push_back(i + 1);
                            }
                        }
                        progress.next();
                    }
                }
            }

            // pack df (col1,col2,val,rank)
            enum
            {
                COL1,
                COL2,
                VAL,
                RANK,
                NCOLS
            };
            const char *NMS[NCOLS] = {"col1", "col2", "val", "rank"};
            SEXP rlist, rnames, rlist_names, rcol1, rcol2, rval, rrank;
            const R_xlen_t N = (R_xlen_t)col1.size();

            rprotect(rlist = RSaneAllocVector(VECSXP, NCOLS));
            rprotect(rlist_names = RSaneAllocVector(STRSXP, NCOLS));
            rprotect(rcol1 = RSaneAllocVector(INTSXP, N));
            rprotect(rcol2 = RSaneAllocVector(INTSXP, N));
            rprotect(rval = RSaneAllocVector(REALSXP, N));
            rprotect(rrank = RSaneAllocVector(INTSXP, N));
            rprotect(rnames = RSaneAllocVector(INTSXP, N));
            for (int i = 0; i < NCOLS; ++i)
                SET_STRING_ELT(rlist_names, i, Rf_mkChar(NMS[i]));
            for (R_xlen_t i = 0; i < N; ++i)
            {
                INTEGER(rcol1)
                [i] = col1[i];
                INTEGER(rcol2)
                [i] = col2[i];
                REAL(rval)
                [i] = val[i];
                INTEGER(rrank)
                [i] = rankv[i];
                INTEGER(rnames)
                [i] = (int)(i + 1);
            }
            // factor levels from X colnames if present
            SEXP xdn = Rf_getAttrib(_x, R_DimNamesSymbol);
            if (!Rf_isNull(xdn) && Rf_xlength(xdn) == 2)
            {
                SEXP levels = VECTOR_ELT(xdn, 1);
                if (!Rf_isNull(levels))
                {
                    Rf_setAttrib(rcol1, R_LevelsSymbol, levels);
                    Rf_setAttrib(rcol1, R_ClassSymbol, Rf_mkString("factor"));
                    Rf_setAttrib(rcol2, R_LevelsSymbol, levels);
                    Rf_setAttrib(rcol2, R_ClassSymbol, Rf_mkString("factor"));
                }
            }
            SET_VECTOR_ELT(rlist, 0, rcol1);
            SET_VECTOR_ELT(rlist, 1, rcol2);
            SET_VECTOR_ELT(rlist, 2, rval);
            SET_VECTOR_ELT(rlist, 3, rrank);
            Rf_setAttrib(rlist, R_NamesSymbol, rlist_names);
            Rf_setAttrib(rlist, R_ClassSymbol, Rf_mkString("data.frame"));
            Rf_setAttrib(rlist, R_RowNamesSymbol, rnames);

            // assign the constructed data.frame to the return value
            ans_df = rlist;
        }
        catch (const std::bad_alloc &)
        {
            rerror("Out of memory");
        }
#if BGNS_HAVE_TGSTAT
        catch (TGLException &e)
        {
            rerror("%s", e.msg());
#else
        catch (const std::exception &e)
        {
            rerror("%s", e.what());
#endif
        }
        rreturn(ans_df);
    }

} // extern "C"