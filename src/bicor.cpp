
#define USE_FC_LEN_T
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <utility>
#include <numeric>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
// Avoid POSIX-only headers to keep Windows portable

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <R.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>
#include <R_ext/Callbacks.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BGNS_ENABLE_SIMD
#define BGNS_ENABLE_SIMD 1
#endif
#ifdef BGNS_STRICT_SCALAR
#undef BGNS_ENABLE_SIMD
#define BGNS_ENABLE_SIMD 0
#endif

#ifndef BGNS_OMP_BUILD_THRESH
#define BGNS_OMP_BUILD_THRESH 8
#endif
#ifndef BGNS_BLOCK_MB
#define BGNS_BLOCK_MB 64
#endif

#if BGNS_ENABLE_SIMD && defined(__AVX2__)
  #include <immintrin.h>
#endif

// ----- light shims (independent of tgstat) -----
struct TGStat { TGStat(SEXP) {} static TGStat &instance() { static TGStat t(R_NilValue); return t; } };
struct ProgressReporter { ProgressReporter(TGStat &, std::uint64_t) {} void next() {} };

#ifndef verror
#define verror(...) Rf_error(__VA_ARGS__)
#endif
#ifndef rerror
#define rerror(...) Rf_error(__VA_ARGS__)
#endif
#ifndef rprotect
#define rprotect(x) do { PROTECT(x); ++__prot_count; } while (0)
#endif
#ifndef rreturn
#define rreturn(x) do { UNPROTECT(__prot_count); return (x); } while (0)
#endif
#ifndef RSaneAllocVector
#define RSaneAllocVector(SEXPTYPE_, N_) Rf_allocVector(SEXPTYPE_, (R_xlen_t)(N_))
#endif

using std::vector; using std::pair; using std::min; using std::max;

namespace {

static inline size_t getenv_mb(const char *name, size_t def_mb) {
    const char *s = std::getenv(name);
    if (!s || !*s) return def_mb;
    long long v = atoll(s);
    if (v < 16) v = 16;
    return (size_t)v;
}
static inline double getenv_double(const char *name, double defv) {
    const char *s = std::getenv(name);
    if (!s || !*s) return defv;
    return atof(s);
}
static inline int getenv_int(const char *name, int defv) {
    const char *s = std::getenv(name);
    if (!s || !*s) return defv;
    int v = atoi(s);
    return (v < 1 ? defv : v);
}
static inline int omp_thresh() { static int c=-1; if(c<0){ c=getenv_int("BGNS_OMP_BUILD_THRESH", BGNS_OMP_BUILD_THRESH); if(c<1)c=1;} return c; }

static inline bool bgns_env_is_false(const char *name) {
    const char *s = std::getenv(name);
    if (!s || !*s) return false;
    while (*s && std::isspace((unsigned char)*s)) ++s;
    if (*s == '0') return true;
    char buf[8]; int i = 0;
    for (; s[i] && i < 7; ++i) buf[i] = (char)std::tolower((unsigned char)s[i]);
    buf[i] = '\0';
    return std::strcmp(buf, "false") == 0 || std::strcmp(buf, "off") == 0 ||
           std::strcmp(buf, "no") == 0;
}

static inline bool bgns_env_is_true(const char *name) {
    const char *s = std::getenv(name);
    if (!s || !*s) return false;
    while (*s && std::isspace((unsigned char)*s)) ++s;
    if (*s == '1') return true;
    char buf[8]; int i = 0;
    for (; s[i] && i < 7; ++i) buf[i] = (char)std::tolower((unsigned char)s[i]);
    buf[i] = '\0';
    return std::strcmp(buf, "true") == 0 || std::strcmp(buf, "on") == 0 ||
           std::strcmp(buf, "yes") == 0;
}

[[maybe_unused]] static inline int bgns_num_threads() {
    int threads = getenv_int("BGNS_NUM_THREADS", 2);
    if (bgns_env_is_true("_R_CHECK_LIMIT_CORES_") && threads > 2) threads = 2;
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (threads > max_threads) threads = max_threads;
#else
    threads = 1;
#endif
    return threads < 1 ? 1 : threads;
}

#ifdef _OPENMP
[[maybe_unused]] static inline int bgns_omp_threads() { return bgns_num_threads(); }
#endif

static void bgns_interrupt_thunk(void *) {
    R_CheckUserInterrupt();
}

static inline void bgns_check_interrupt() {
    if (R_ToplevelExec(bgns_interrupt_thunk, nullptr) == FALSE) {
        Rf_error("Interrupted");
    }
}

template <class Fn>
static inline void bgns_parallel_for(int n, Fn&& body, int min_parallel = -1) {
    if (min_parallel < 0) min_parallel = omp_thresh();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= min_parallel) num_threads(bgns_omp_threads())
#endif
    for (int i = 0; i < n; ++i) body(i);
}

enum class BgnsRuntimeIsa : int { scalar = 0, avx2 = 1, neon = 2 };

[[maybe_unused]] static inline BgnsRuntimeIsa bgns_detect_runtime_isa() {
    // C++ static initialization is synchronized when panel workers first
    // reach the optional SIMD kernel concurrently.
    static const BgnsRuntimeIsa cached = [] {
        if (bgns_env_is_false("BGNS_SIMD")) return BgnsRuntimeIsa::scalar;
#if defined(__aarch64__) || defined(_M_ARM64)
        return BgnsRuntimeIsa::neon;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
# if defined(__GNUC__) && !defined(__clang__)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2")) return BgnsRuntimeIsa::avx2;
# elif defined(__AVX2__)
        return BgnsRuntimeIsa::avx2;
# endif
#endif
        return BgnsRuntimeIsa::scalar;
    }();
    return cached;
}

[[maybe_unused]] static inline bool bgns_runtime_allows_avx2() {
#if BGNS_ENABLE_SIMD && defined(__AVX2__)
    return bgns_detect_runtime_isa() == BgnsRuntimeIsa::avx2;
#else
    return false;
#endif
}

struct BgnsBlock { int begin; int end; };
[[maybe_unused]] static inline int bgns_block_count(int n, int block) { return (n <= 0 || block <= 0) ? 0 : (n + block - 1) / block; }
[[maybe_unused]] static inline BgnsBlock bgns_block_at(int n, int block, int b) {
    const int begin = b * block;
    const int end = begin + block < n ? begin + block : n;
    return {begin, end};
}

struct TLS {
    vector<double> tmp, tmp2, absbuf;
    vector<int>    tmpi;
};
static inline TLS& tls() { static thread_local TLS t; return t; }

struct Bitset {
    vector<uint64_t> w;
    inline void init_zero(int n){ w.assign(((size_t)n+63)>>6, 0ull); }
    inline void set(int i){ w[(size_t)i>>6] |= (1ull << (i & 63)); }
    static inline int popcnt64(uint64_t x){
    #if defined(_MSC_VER) && defined(_M_X64)
        return (int)__popcnt64(x);
    #else
        return __builtin_popcountll(x);
    #endif
    }
    inline int popcnt_and(const Bitset &o) const {
        const size_t m = std::min(w.size(), o.w.size());
        int s=0; for(size_t i=0;i<m;++i) s += popcnt64(w[i] & o.w[i]); return s;
    }
};

inline double median_inplace(vector<double> &v) {
    const size_t n=v.size(); if(!n) return NA_REAL; const size_t mid=n/2;
    std::nth_element(v.begin(), v.begin()+mid, v.end());
    double m=v[mid]; if((n&1U)==0U){ auto it=std::max_element(v.begin(), v.begin()+mid); m = (std::fabs(m) > std::numeric_limits<double>::max()/2 || std::fabs(*it) > std::numeric_limits<double>::max()/2) ? 0.5*m + 0.5*(*it) : 0.5*(m+*it); }
    return m;
}
inline double mad_raw_tls_from_vec(const vector<double> &vals, double med){
    if(vals.empty()) return NA_REAL; TLS &t=tls(); t.absbuf.resize(vals.size());
    for(size_t i=0;i<vals.size();++i) t.absbuf[i]=std::fabs(vals[i]-med);
    double u = median_inplace(t.absbuf); if(!R_FINITE(u) || u<=0.0) return NA_REAL; return u;
}

#if BGNS_ENABLE_SIMD && defined(__AVX2__)
static inline __m256d mm256_abs_pd(__m256d x){ const __m256i mask=_mm256_set1_epi64x(0x7fffffffffffffffLL); return _mm256_and_pd(x, _mm256_castsi256_pd(mask)); }
#endif

// Store weighted standardized deviations. Each column differs from the
// unscaled midvariate by one positive constant, which cancels in both
// full-column and overlap-specific correlation denominators.
inline double bounded_midvariate(double x, double med, double mad, double inv_k) {
    const double diff = x - med;
    const double tuning = 9.0 * 1.4826;
    const double u = !R_FINITE(diff) ? (x/mad - med/mad)/tuning :
        ((R_FINITE(inv_k) && inv_k > 0.0) ? diff*inv_k : (diff/mad)/tuning);
    if (!(std::fabs(u) < 1.0)) return 0.0;
    const double w = 1.0 - u*u;
    return u*w*w;
}

inline void midvariates_dense_compute(const double *vals, int n, double med, double mad,
                                      double *out_mv, double &out_ss){
    const double inv_k = (1.0 / (9.0 * 1.4826)) / mad;
    out_ss=0.0;
    int i=0;
#if BGNS_ENABLE_SIMD && defined(__AVX2__)
    if (bgns_runtime_allows_avx2() && R_FINITE(inv_k) && inv_k > 0.0) {
        const int n4 = n & ~3;
        const __m256d v_med=_mm256_set1_pd(med), v_invk=_mm256_set1_pd(inv_k);
        const __m256d vone=_mm256_set1_pd(1.0), vmax=_mm256_set1_pd(std::numeric_limits<double>::max());
        for(;i<n4;i+=4){
            const __m256d diff=_mm256_sub_pd(_mm256_loadu_pd(vals+i), v_med);
            if (_mm256_movemask_pd(_mm256_cmp_pd(mm256_abs_pd(diff), vmax, _CMP_GT_OQ))) {
                for (int q=0;q<4;++q) {
                    const double mv=bounded_midvariate(vals[i+q],med,mad,inv_k);
                    out_mv[i+q]=mv; out_ss+=mv*mv;
                }
                continue;
            }
            const __m256d u=_mm256_mul_pd(diff,v_invk);
            const __m256d inside=_mm256_cmp_pd(mm256_abs_pd(u),vone,_CMP_LT_OQ);
            const __m256d t=_mm256_sub_pd(vone,_mm256_mul_pd(u,u));
            const __m256d mv=_mm256_and_pd(inside,_mm256_mul_pd(u,_mm256_mul_pd(t,t)));
            _mm256_storeu_pd(out_mv+i,mv);
            alignas(32) double tmp[4]; _mm256_store_pd(tmp,mv);
            out_ss+=tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]+tmp[3]*tmp[3];
        }
    }
#endif
    for(;i<n;++i){
        const double mv=bounded_midvariate(vals[i],med,mad,inv_k);
        out_mv[i]=mv; out_ss+=mv*mv;
    }
}

struct DenseCol{ vector<double> mv; double sumsq=0; bool ok=false; };
inline DenseCol build_midcol_dense(const double *col, int n){
    DenseCol out; TLS &t=tls(); t.tmp.resize(n);
    for(int i=0;i<n;++i){ double v=col[i]; if(!R_FINITE(v)){ out.ok=false; return out; } t.tmp[i]=v; }
    t.tmp2=t.tmp; double med=median_inplace(t.tmp2); double mad=mad_raw_tls_from_vec(t.tmp, med);
    if(!(mad>0.0)){ out.ok=false; return out; }
    out.mv.resize(n); double ss=0.0;
    midvariates_dense_compute(t.tmp.data(), n, med, mad, out.mv.data(), ss);
    out.sumsq=ss; out.ok=(ss>0.0); return out;
}

struct DotStat{ double num, sA, sB; int overlap; };
inline double dot_dense_dense_num(const double *a, const double *b, int n){ double s=0.0; for(int i=0;i<n;++i) s+=a[i]*b[i]; return s; }

template<typename IA, typename IB>
inline DotStat gallop_num(const IA* ai, const double* av, int na, const IB* bi, const double* bv, int nb){
    double num=0.0; int ov=0; int jb=0;
    for(int ia=0; ia<na; ++ia){ int key=(int)ai[ia];
        jb = (int)(std::lower_bound(bi+jb, bi+nb, (IB)key) - bi);
        if(jb==nb) break; if((int)bi[jb]==key){ num += av[ia]*bv[jb]; ++ov; }
    }
    return {num,0.0,0.0,ov};
}
template<typename IA, typename IB>
inline DotStat gallop_inter(const IA* ai, const double* av, int na, const IB* bi, const double* bv, int nb){
    double num=0.0, sA=0.0, sB=0.0; int ov=0; int jb=0;
    for(int ia=0; ia<na; ++ia){ int key=(int)ai[ia];
        jb = (int)(std::lower_bound(bi+jb, bi+nb, (IB)key) - bi);
        if(jb==nb) break; if((int)bi[jb]==key){ double a=av[ia], b=bv[jb]; num+=a*b; sA+=a*a; sB+=b*b; ++ov; }
    }
    return {num,sA,sB,ov};
}

template<typename I>
inline DotStat full_sparse_num(const double *mvF, const I *idxS, const double *mvS, int k){
    double num=0.0; for(int i=0;i<k;++i) num += mvF[(int)idxS[i]] * mvS[i]; return {num,0.0,0.0,k};
}
template<typename I>
inline DotStat full_sparse_inter(const double *mvF, const I *idxS, const double *mvS, int k, double sumsqS){
    double num=0.0, sA=0.0; for(int i=0;i<k;++i){ double a=mvF[(int)idxS[i]], b=mvS[i]; num+=a*b; sA+=a*a; }
    return {num, sA, sumsqS, k};
}

inline void dgemm_AB(char tA, char tB, const int M, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb, const double beta, double *C, const int ldc){
    F77_CALL(dgemm)(&tA, &tB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc FCONE FCONE);
}

static inline void dense_panel_sizes(int n, int px, int py, size_t mem_mb, int &Pa, int &Pb){
    const double S=(double)mem_mb*1024.0*1024.0;
    const double p_sym = std::floor(std::max(1.0, std::sqrt((double)n*(double)n + S/8.0) - (double)n));
    Pa=(int)std::max(1.0, std::min(p_sym, (double)px));
    Pb=(int)std::max(1.0, std::min(p_sym, (double)py));
    if(Pa<1)Pa=1; if(Pb<1)Pb=1;
}
template<typename I>
static inline int sparse_panel_cols(int n, int p_total, size_t mem_mb){
    const double S=(double)mem_mb*1024.0*1024.0;
    const double per_col = 8.0*n + (double)sizeof(I)*n + 0.125*n + 256.0;
    int pc = (int)std::max(1.0, std::min((double)p_total, std::floor(S / per_col)));
    return pc>0?pc:1;
}
template<typename I>
static inline bool should_build_bitset(int n, int k){
    const double beta = getenv_double("BGNS_BITSET_BETA", 0.75);
    return ((double)n*0.125) <= beta * (double)k * (double)sizeof(I);
}

struct PanelDense {
    int n=0;
    vector<int> cols; vector<double> sumsq; vector<char> ok; vector<double> mv;
    inline const double* mv_col_ptr(int d) const { return mv.data() + (size_t)d*(size_t)n; }
    inline double*       mv_col_ptr(int d)       { return mv.data() + (size_t)d*(size_t)n; }
};
template<typename I>
struct PanelSparse {
    int n=0; vector<int> cols; vector<size_t> off; vector<I> idx; vector<double> mv; vector<double> sumsq; vector<char> ok, has_bits; vector<Bitset> bits;
    inline int nnz_col(int s) const { return (int)(off[s+1]-off[s]); }
    inline const I* idx_ptr(int s) const { return idx.data()+off[s]; }
    inline const double* mv_ptr(int s) const { return mv.data()+off[s]; }
};
template<typename I>
struct PanelMixed {
    int n=0, P=0; vector<int> local_cols; vector<char> is_dense; vector<int> map_dense, map_sparse; vector<char> ok_local; PanelDense D; PanelSparse<I> S;
};

[[maybe_unused]] static inline bool in_upper_triangle(int gi, int gj){ return gi < gj; }

static inline bool chr_eq(SEXP a, SEXP b){
    if(a==b) return true; if(a==NA_STRING && b==NA_STRING) return true; if(a==NA_STRING || b==NA_STRING) return false; return std::strcmp(CHAR(a), CHAR(b))==0;
}
static inline bool stringvec_identical(SEXP A, SEXP B){
    if(!Rf_isString(A) || !Rf_isString(B)) return false; const R_xlen_t n=Rf_xlength(A); if(n!=Rf_xlength(B)) return false;
    for(R_xlen_t i=0;i<n;++i){ if(!chr_eq(STRING_ELT(A,i), STRING_ELT(B,i))) return false; } return true;
}

// Builders for dense/int base matrices
static inline void build_dense_panel_X_real(const double *x, int n, const int *cols, int Pa, vector<double> &A, vector<double> &ssX){
    A.assign((size_t)n*(size_t)Pa, 0.0); ssX.assign(Pa, 0.0);
    bgns_parallel_for(Pa, [&](int j){ DenseCol dc = build_midcol_dense(x + (size_t)cols[j]*(size_t)n, n);
        ssX[j]=dc.ok?dc.sumsq:0.0; if(dc.ok){ double *dst=A.data()+(size_t)j*(size_t)n; for(int i=0;i<n;++i) dst[i]=dc.mv[i]; } });
}
static inline void build_dense_panel_X_int(const int *x, int n, const int *cols, int Pa, vector<double> &A, vector<double> &ssX){
    A.assign((size_t)n*(size_t)Pa, 0.0); ssX.assign(Pa, 0.0);
    bgns_parallel_for(Pa, [&](int j){ TLS &t=tls(); t.tmp.resize(n); bool ok=true;
        for(int i=0;i<n;++i){ int v = x[(size_t)cols[j]*(size_t)n + i]; if(v==NA_INTEGER){ ok=false; break; } t.tmp[i]=(double)v; }
        if(!ok){ ssX[j]=0.0; return; } t.tmp2=t.tmp; double med=median_inplace(t.tmp2); double mad=mad_raw_tls_from_vec(t.tmp, med); if(!(mad>0.0)){ ssX[j]=0.0; return; }
        double *dst=A.data()+(size_t)j*(size_t)n; double ss=0.0;  midvariates_dense_compute(t.tmp.data(), n, med, mad, dst, ss); ssX[j]=ss; });
}

// CSC helpers
struct CSCRef { SEXP pS,iS,xS,dimS,dnS; const int *p,*i; const double *x; int nrow,ncol; };
static inline bool load_csc(SEXP obj, CSCRef &r){
    if(!(Rf_isS4(obj) && Rf_inherits(obj,"dgCMatrix"))) return false;
    r.pS=R_do_slot(obj,Rf_install("p")); r.iS=R_do_slot(obj,Rf_install("i")); r.xS=R_do_slot(obj,Rf_install("x")); r.dimS=R_do_slot(obj,Rf_install("Dim")); r.dnS=R_do_slot(obj,Rf_install("Dimnames"));
    if(!Rf_isInteger(r.pS)||!Rf_isInteger(r.iS)||!Rf_isReal(r.xS)||!Rf_isInteger(r.dimS)) verror("dgCMatrix slots have unexpected types");
    r.p=INTEGER(r.pS); r.i=INTEGER(r.iS); r.x=REAL(r.xS); if(Rf_xlength(r.dimS)!=2) verror("Dim length mismatch"); r.nrow=INTEGER(r.dimS)[0]; r.ncol=INTEGER(r.dimS)[1]; return true;
}
struct CscColumnSummary {
    int stored = 0;
    int finite = 0;
    int structural_zeros = 0;
    double sum = 0.0;
    double sumsq = 0.0;
    bool has_nonfinite = false;
};

static inline CscColumnSummary summarize_csc_column(const CSCRef &M, int col) {
    CscColumnSummary s;
    const int st = M.p[col], ed = M.p[col + 1];
    s.stored = ed - st;
    s.structural_zeros = M.nrow - s.stored;
    if (s.structural_zeros < 0) s.structural_zeros = 0;
    for (int k = st; k < ed; ++k) {
        const double v = M.x[k];
        if (R_FINITE(v)) { ++s.finite; s.sum += v; s.sumsq += v * v; }
        else s.has_nonfinite = true;
    }
    return s;
}

static inline bool csc_all_finite(const CSCRef &M){
    for (int j = 0; j < M.ncol; ++j) if (summarize_csc_column(M, j).has_nonfinite) return false;
    return true;
}
static inline SEXP get_colnames_any(SEXP obj, int p){
    if(Rf_isMatrix(obj)){ SEXP dn=Rf_getAttrib(obj,R_DimNamesSymbol); if(!Rf_isNull(dn)&&Rf_xlength(dn)==2){ SEXP cn=VECTOR_ELT(dn,1); if(!Rf_isNull(cn)&&Rf_isString(cn)&&Rf_xlength(cn)==(R_xlen_t)p) return cn; } return R_NilValue; }
    if(Rf_isS4(obj)&&Rf_inherits(obj,"dgCMatrix")){ SEXP dn=R_do_slot(obj,Rf_install("Dimnames")); if(!Rf_isNull(dn)&&Rf_xlength(dn)==2){ SEXP cn=VECTOR_ELT(dn,1); if(!Rf_isNull(cn)&&Rf_isString(cn)&&Rf_xlength(cn)==(R_xlen_t)p) return cn; } }
    return R_NilValue;
}
static inline void build_dense_panel_X_csc(const int *Cp, const int *Ci, const double *Cx, int n, const int *cols, int Pa, vector<double> &A, vector<double> &ssX){
    A.assign((size_t)n*(size_t)Pa, 0.0); ssX.assign(Pa, 0.0);
    bgns_parallel_for(Pa, [&](int j){
        int col=cols[j]; TLS &t=tls(); t.tmp.assign(n, 0.0); bool ok=true; int st=Cp[col], ed=Cp[col+1];
        for(int k=st;k<ed;++k){ int row=Ci[k]; double v=Cx[k]; if(!R_FINITE(v)){ ok=false; break; } t.tmp[row]=v; }
        if(!ok){ ssX[j]=0.0; return; } t.tmp2=t.tmp; double med=median_inplace(t.tmp2); double mad=mad_raw_tls_from_vec(t.tmp, med); if(!(mad>0.0)){ ssX[j]=0.0; return; }
        double *dst=A.data()+(size_t)j*(size_t)n; double ss=0.0;  midvariates_dense_compute(t.tmp.data(), n, med, mad, dst, ss); ssX[j]=ss;
    });
}

template<typename I>
static inline void build_panel_mixed_from_real(const double *base, int n, const int *cols, int P, PanelMixed<I> &pm){
    pm.n=n; pm.P=P; pm.local_cols.assign(cols, cols+P); pm.is_dense.assign(P,0); pm.map_dense.assign(P,-1); pm.map_sparse.assign(P,-1); pm.ok_local.assign(P,0);
    pm.D.n=n; pm.S.n=n; pm.S.off.clear(); pm.S.off.push_back(0);
    for(int lp=0; lp<P; ++lp){
        int gj=cols[lp]; const double *col = base + (size_t)gj*(size_t)n; TLS &t=tls(); t.tmp.clear(); t.tmp.reserve(n); t.tmpi.clear(); t.tmpi.reserve(n);
        for(int i=0;i<n;++i){ double v=col[i]; if(R_FINITE(v)){ t.tmpi.push_back(i); t.tmp.push_back(v); } }
        if(t.tmp.empty()){ pm.ok_local[lp]=0; continue; }
        t.tmp2=t.tmp; double med=median_inplace(t.tmp2); double mad=mad_raw_tls_from_vec(t.tmp, med); if(!(mad>0.0)){ pm.ok_local[lp]=0; continue; }
         int m=(int)t.tmp.size();
        if(m==n){ int dpos=(int)pm.D.cols.size(); pm.is_dense[lp]=1; pm.map_dense[lp]=dpos; pm.ok_local[lp]=1; pm.D.cols.push_back(gj); pm.D.sumsq.push_back(0.0); pm.D.ok.push_back(1);
            size_t old=pm.D.mv.size(); pm.D.mv.resize(old+(size_t)n); double *out=pm.D.mv.data()+old; double ss=0.0; midvariates_dense_compute(t.tmp.data(), n, med, mad, out, ss); pm.D.sumsq.back()=ss;
        } else {
            int spos=(int)pm.S.cols.size(); pm.is_dense[lp]=0; pm.map_sparse[lp]=spos; pm.ok_local[lp]=1; pm.S.cols.push_back(gj); pm.S.sumsq.push_back(0.0); pm.S.ok.push_back(1);
            size_t base_off=pm.S.idx.size(); pm.S.idx.resize(base_off+(size_t)m); pm.S.mv.resize(base_off+(size_t)m);
            double ss=0.0; midvariates_dense_compute(t.tmp.data(), m, med, mad, pm.S.mv.data()+base_off, ss); pm.S.sumsq.back()=ss;
            for(int q=0;q<m;++q) pm.S.idx[base_off+q]=(I)t.tmpi[q]; pm.S.off.push_back(base_off+(size_t)m);
            if(should_build_bitset<I>(n,m)){ pm.S.has_bits.push_back(1); pm.S.bits.emplace_back(); pm.S.bits.back().init_zero(n); for(int q=0;q<m;++q) pm.S.bits.back().set(t.tmpi[q]); }
            else { pm.S.has_bits.push_back(0); pm.S.bits.emplace_back(); }
        }
    }
}
template<typename I>
static inline void build_panel_mixed_from_int(const int *base, int n, const int *cols, int P, PanelMixed<I> &pm){
    pm.n=n; pm.P=P; pm.local_cols.assign(cols, cols+P); pm.is_dense.assign(P,0); pm.map_dense.assign(P,-1); pm.map_sparse.assign(P,-1); pm.ok_local.assign(P,0);
    pm.D.n=n; pm.S.n=n; pm.S.off.clear(); pm.S.off.push_back(0);
    for(int lp=0; lp<P; ++lp){
        int gj=cols[lp]; const int *col = base + (size_t)gj*(size_t)n; TLS &t=tls(); t.tmp.clear(); t.tmpi.clear(); t.tmp.reserve(n); t.tmpi.reserve(n);
        for(int i=0;i<n;++i){ int v=col[i]; if(v!=NA_INTEGER){ t.tmpi.push_back(i); t.tmp.push_back((double)v); } }
        if(t.tmp.empty()){ pm.ok_local[lp]=0; continue; }
        t.tmp2=t.tmp; double med=median_inplace(t.tmp2); double mad=mad_raw_tls_from_vec(t.tmp, med); if(!(mad>0.0)){ pm.ok_local[lp]=0; continue; }
         int m=(int)t.tmp.size();
        if(m==n){ int dpos=(int)pm.D.cols.size(); pm.is_dense[lp]=1; pm.map_dense[lp]=dpos; pm.ok_local[lp]=1; pm.D.cols.push_back(gj); pm.D.sumsq.push_back(0.0); pm.D.ok.push_back(1);
            size_t old=pm.D.mv.size(); pm.D.mv.resize(old+(size_t)n); double *out=pm.D.mv.data()+old; double ss=0.0; midvariates_dense_compute(t.tmp.data(), n, med, mad, out, ss); pm.D.sumsq.back()=ss;
        } else {
            int spos=(int)pm.S.cols.size(); pm.is_dense[lp]=0; pm.map_sparse[lp]=spos; pm.ok_local[lp]=1; pm.S.cols.push_back(gj); pm.S.sumsq.push_back(0.0); pm.S.ok.push_back(1);
            size_t base_off=pm.S.idx.size(); pm.S.idx.resize(base_off+(size_t)m); pm.S.mv.resize(base_off+(size_t)m);
            double ss=0.0; midvariates_dense_compute(t.tmp.data(), m, med, mad, pm.S.mv.data()+base_off, ss); pm.S.sumsq.back()=ss;
            for(int q=0;q<m;++q) pm.S.idx[base_off+q]=(I)t.tmpi[q]; pm.S.off.push_back(base_off+(size_t)m);
            if(should_build_bitset<I>(n,m)){ pm.S.has_bits.push_back(1); pm.S.bits.emplace_back(); pm.S.bits.back().init_zero(n); for(int q=0;q<m;++q) pm.S.bits.back().set(t.tmpi[q]); }
            else { pm.S.has_bits.push_back(0); pm.S.bits.emplace_back(); }
        }
    }
}
template<typename I>
static inline void build_panel_mixed_from_csc(const int *Cp, const int *Ci, const double *Cx, int n, const int *cols, int P, PanelMixed<I> &pm){
    pm.n=n; pm.P=P; pm.local_cols.assign(cols, cols+P); pm.is_dense.assign(P,0); pm.map_dense.assign(P,-1); pm.map_sparse.assign(P,-1); pm.ok_local.assign(P,0);
    pm.D.n=n; pm.S.n=n; pm.S.off.clear(); pm.S.off.push_back(0);
    for(int lp=0; lp<P; ++lp){
        int gj=cols[lp]; TLS &t=tls(); t.tmp.assign(n, 0.0); t.tmpi.clear(); t.tmp2.clear(); bool any_finite=false;
        int st=Cp[gj], ed=Cp[gj+1]; for(int k=st;k<ed;++k){ int row=Ci[k]; double v=Cx[k]; if(!R_FINITE(v)) { /* NA remains non-finite; excluded */ } t.tmp[row]=v; }
        for(int r=0;r<n;++r){ double v=t.tmp[r]; if(R_FINITE(v)){ t.tmpi.push_back(r); t.tmp2.push_back(v); any_finite=true; } }
        if(!any_finite){ pm.ok_local[lp]=0; continue; }
        TLS &t2=tls(); t2.tmp2=t.tmp2; double med=median_inplace(t2.tmp2); double mad=mad_raw_tls_from_vec(t.tmp2, med); if(!(mad>0.0)){ pm.ok_local[lp]=0; continue; }
         int m=(int)t.tmp2.size();
        if(m==n){ int dpos=(int)pm.D.cols.size(); pm.is_dense[lp]=1; pm.map_dense[lp]=dpos; pm.ok_local[lp]=1; pm.D.cols.push_back(gj); pm.D.sumsq.push_back(0.0); pm.D.ok.push_back(1);
            size_t old=pm.D.mv.size(); pm.D.mv.resize(old+(size_t)n); double *out=pm.D.mv.data()+old; double ss=0.0; midvariates_dense_compute(t.tmp.data(), n, med, mad, out, ss); pm.D.sumsq.back()=ss;
        } else {
            int spos=(int)pm.S.cols.size(); pm.is_dense[lp]=0; pm.map_sparse[lp]=spos; pm.ok_local[lp]=1; pm.S.cols.push_back(gj); pm.S.sumsq.push_back(0.0); pm.S.ok.push_back(1);
            size_t base_off=pm.S.idx.size(); pm.S.idx.resize(base_off+(size_t)m); pm.S.mv.resize(base_off+(size_t)m);
            // median_inplace() above permutes t.tmp2. Restore finite values in
            // row-index order before computing midvariates so pm.S.mv[q]
            // remains aligned with pm.S.idx[q].
            for(int q=0;q<m;++q) t.tmp2[q]=t.tmp[t.tmpi[q]];
            double ss=0.0; midvariates_dense_compute(t.tmp2.data(), m, med, mad, pm.S.mv.data()+base_off, ss); pm.S.sumsq.back()=ss;
            for(int q=0;q<m;++q) pm.S.idx[base_off+q]=(I)t.tmpi[q]; pm.S.off.push_back(base_off+(size_t)m);
            if(should_build_bitset<I>(n,m)){ pm.S.has_bits.push_back(1); pm.S.bits.emplace_back(); pm.S.bits.back().init_zero(n); for(int q=0;q<m;++q) pm.S.bits.back().set(t.tmpi[q]); }
            else { pm.S.has_bits.push_back(0); pm.S.bits.emplace_back(); }
        }
    }
}


template<typename I>
static inline void build_panel_mixed_any(SEXP obj, bool is_mat, const CSCRef *csc, int n, const int *cols, int P, PanelMixed<I> &pm){
    if (is_mat) {
        if (Rf_isReal(obj)) build_panel_mixed_from_real<I>(REAL(obj), n, cols, P, pm);
        else                build_panel_mixed_from_int <I>(INTEGER(obj), n, cols, P, pm);
    } else {
        build_panel_mixed_from_csc<I>(csc->p, csc->i, csc->x, n, cols, P, pm);
    }
}

template<typename I>
static inline double panel_mixed_bicor_pair(const PanelMixed<I> &PX, int si,
                                            const PanelMixed<I> &PY, int tj,
                                            bool pairwise_complete,
                                            bool use_inter_den,
                                            int min_overlap){
    if (!PX.ok_local[si] || !PY.ok_local[tj]) return NA_REAL;
    const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
    if (!pairwise_complete && !(Sd && Td)) return NA_REAL;

    double r = NA_REAL;
    if (Sd && Td) {
        if (PX.n < min_overlap) return NA_REAL;
        const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
        const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
        const double ss_s = PX.D.sumsq[PX.map_dense[si]];
        const double ss_t = PY.D.sumsq[PY.map_dense[tj]];
        if (ss_s > 0.0 && ss_t > 0.0) {
            const double num = dot_dense_dense_num(a, b, PX.n);
            const double den = std::sqrt(ss_s * ss_t);
            if (den > 0.0) r = num / den;
        }
    } else if (Sd && !Td) {
        const int tpos = PY.map_sparse[tj];
        const int k = PY.S.nnz_col(tpos);
        if (k >= min_overlap) {
            const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
            if (use_inter_den) {
                auto st = full_sparse_inter(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k, PY.S.sumsq[tpos]);
                const double den = std::sqrt(st.sA * st.sB);
                if (den > 0.0) r = st.num / den;
            } else {
                auto st = full_sparse_num(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k);
                const double den = std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[tpos]);
                if (den > 0.0) r = st.num / den;
            }
        }
    } else if (!Sd && Td) {
        const int spos = PX.map_sparse[si];
        const int k = PX.S.nnz_col(spos);
        if (k >= min_overlap) {
            const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
            if (use_inter_den) {
                auto st = full_sparse_inter(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k, PX.S.sumsq[spos]);
                const double den = std::sqrt(st.sA * st.sB);
                if (den > 0.0) r = st.num / den;
            } else {
                auto st = full_sparse_num(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k);
                const double den = std::sqrt(PX.S.sumsq[spos] * PY.D.sumsq[PY.map_dense[tj]]);
                if (den > 0.0) r = st.num / den;
            }
        }
    } else {
        const int spos = PX.map_sparse[si];
        const int tpos = PY.map_sparse[tj];
        const int ks = PX.S.nnz_col(spos);
        const int kt = PY.S.nnz_col(tpos);
        if (ks >= min_overlap && kt >= min_overlap) {
            bool ok = true;
            if (PX.S.has_bits[spos] && PY.S.has_bits[tpos]) {
                const int ov = PX.S.bits[spos].popcnt_and(PY.S.bits[tpos]);
                ok = (ov >= min_overlap);
            }
            if (ok) {
                if (use_inter_den) {
                    auto st = gallop_inter(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), ks,
                                           PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kt);
                    if (st.overlap >= min_overlap) {
                        const double den = std::sqrt(st.sA * st.sB);
                        if (den > 0.0) r = st.num / den;
                    }
                } else {
                    auto st = gallop_num(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), ks,
                                         PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kt);
                    if (st.overlap >= min_overlap) {
                        const double den = std::sqrt(PX.S.sumsq[spos] * PY.S.sumsq[tpos]);
                        if (den > 0.0) r = st.num / den;
                    }
                }
            }
        }
    }
    return r;
}

static inline bool topk_pair_better(const pair<int,double>& a, const pair<int,double>& b){
    return a.second > b.second || (a.second == b.second && a.first < b.first);
}

static inline void shrink_topk_buffer(vector<pair<int,double>> &buf, int K){
    if (K < 1) { buf.clear(); return; }
    const size_t k = static_cast<size_t>(K);
    if (k <= std::numeric_limits<size_t>::max()/4 && buf.size() > 4*k) {
        std::nth_element(buf.begin(), buf.begin() + (2*k - 1), buf.end(), topk_pair_better);
        buf.resize(2*k);
    }
}

static inline void tiny_topk_consider(vector<pair<int,double>> &buf, int K, int idx, double score, double threshold){
    if (!(R_FINITE(score) && score > threshold)) return;
    buf.emplace_back(idx, score);
    shrink_topk_buffer(buf, K);
}

static inline void tiny_topk_finalize(vector<pair<int,double>> &buf, int K){
    if (K < 1) { buf.clear(); return; }
    if (buf.empty()) return;
    const int keep = (int)std::min((size_t)K, buf.size());
    std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(), topk_pair_better);
    std::sort(buf.begin(), buf.begin()+keep, topk_pair_better);
    buf.resize(keep);
}

static inline void append_final_topk(const vector<pair<int,double>> &src_buf, int target_col0, int K,
                                     vector<int> &col1, vector<int> &col2,
                                     vector<double> &val, vector<int> &rankv){
    if (src_buf.empty()) return;
    vector<pair<int,double>> buf(src_buf);
    tiny_topk_finalize(buf, K);
    for (int k=0; k<(int)buf.size(); ++k) {
        col1.push_back(target_col0 + 1);
        col2.push_back(buf[k].first + 1);
        val.push_back(buf[k].second);
        rankv.push_back(k + 1);
    }
}

static inline bool knn_use_separate_levels(){
    const char *mode_env = std::getenv("BGNS_KNN_BIPARTITE_LEVELS");
    return mode_env && (
        std::strcmp(mode_env,"separate")==0 ||
        std::strcmp(mode_env,"true")==0 ||
        std::strcmp(mode_env,"1")==0 ||
        std::strcmp(mode_env,"dual")==0 ||
        std::strcmp(mode_env,"yx")==0
    );
}

static inline void set_knn_factor_levels(SEXP rcol1, SEXP rcol2, SEXP _x, int px, SEXP _y, int py, bool haveY){
    SEXP xlev = get_colnames_any(_x, px);
    SEXP ylev = haveY ? get_colnames_any(_y, py) : R_NilValue;
    const bool use_separate = knn_use_separate_levels();

    if (!haveY) {
        if (!Rf_isNull(xlev)) {
            Rf_setAttrib(rcol1, R_LevelsSymbol, xlev); Rf_setAttrib(rcol1, R_ClassSymbol, Rf_mkString("factor"));
            Rf_setAttrib(rcol2, R_LevelsSymbol, xlev); Rf_setAttrib(rcol2, R_ClassSymbol, Rf_mkString("factor"));
        }
    } else if (use_separate) {
        if (!Rf_isNull(ylev)) { Rf_setAttrib(rcol1, R_LevelsSymbol, ylev); Rf_setAttrib(rcol1, R_ClassSymbol, Rf_mkString("factor")); }
        if (!Rf_isNull(xlev)) { Rf_setAttrib(rcol2, R_LevelsSymbol, xlev); Rf_setAttrib(rcol2, R_ClassSymbol, Rf_mkString("factor")); }
    } else {
        if (!Rf_isNull(xlev) && !Rf_isNull(ylev) && (px==py) && stringvec_identical(xlev, ylev)) {
            Rf_setAttrib(rcol1, R_LevelsSymbol, xlev); Rf_setAttrib(rcol1, R_ClassSymbol, Rf_mkString("factor"));
            Rf_setAttrib(rcol2, R_LevelsSymbol, xlev); Rf_setAttrib(rcol2, R_ClassSymbol, Rf_mkString("factor"));
        } else if (Rf_isNull(xlev) && Rf_isNull(ylev)) {
            // keep as integers
        } else {
            verror("bicor_knn: bipartite inputs have different or missing column-name levels.\nUse bipartite_levels = \"separate\" to allow distinct factor levels.");
        }
    }
}

[[maybe_unused]] static inline bool in_upper_triangle(int gi, int gj, bool strict){ return gi < gj || (!strict && gi==gj); }

} // namespace

extern "C" {

// ====== Dense matrix bicor (full matrix result) ======
SEXP C_bicor(SEXP _x, SEXP _y, SEXP _pairwise, SEXP _use_inter_den, SEXP _envir)
{
    SEXP answer = R_NilValue; int __prot_count = 0;
    try{
        TGStat tgstat(_envir);
        const bool haveY = !Rf_isNull(_y);
        if ((!Rf_isReal(_x) && !Rf_isInteger(_x)) || (haveY && !Rf_isReal(_y) && !Rf_isInteger(_y))) verror("\"x\"/\"y\" must be numeric matrices");
        if (!Rf_isMatrix(_x) || (haveY && !Rf_isMatrix(_y))) verror("\"x\"/\"y\" must be matrices");
        const int n=Rf_nrows(_x), px=Rf_ncols(_x), py=haveY?Rf_ncols(_y):px;
        if (haveY && Rf_nrows(_y) != n) verror("x and y must have same number of rows");
        const bool pairwise_complete = Rf_asLogical(_pairwise);
        const bool use_inter_den     = Rf_asLogical(_use_inter_den);
        const int  min_overlap       = getenv_int("BGNS_MIN_OVERLAP", 3);

        bool na_free_x=true, na_free_y=true;
        if (Rf_isReal(_x)) { const double *xp=REAL(_x); for(int j=0;j<px && na_free_x;++j) for(int i=0;i<n;++i) if(!R_FINITE(xp[(size_t)j*n+i])){ na_free_x=false; break; } }
        else { const int *xi=INTEGER(_x); for(int j=0;j<px && na_free_x;++j) for(int i=0;i<n;++i) if(xi[(size_t)j*n+i]==NA_INTEGER){ na_free_x=false; break; } }
        if (haveY){
            if (Rf_isReal(_y)) { const double *yp=REAL(_y); for(int j=0;j<py && na_free_y;++j) for(int i=0;i<n;++i) if(!R_FINITE(yp[(size_t)j*n+i])){ na_free_y=false; break; } }
            else { const int *yi=INTEGER(_y); for(int j=0;j<py && na_free_y;++j) for(int i=0;i<n;++i) if(yi[(size_t)j*n+i]==NA_INTEGER){ na_free_y=false; break; } }
        }
        const bool na_free = na_free_x && (!haveY || na_free_y);

        rprotect(answer = RSaneAllocVector(REALSXP, (uint64_t)px*(uint64_t)py));
        double *ans = REAL(answer); for(uint64_t t=0,T=(uint64_t)px*(uint64_t)py; t<T; ++t) ans[t]=NA_REAL;
        ProgressReporter progress(TGStat::instance(), haveY ? (uint64_t)py : (uint64_t)px);

        const bool use_blas = na_free && !use_inter_den && n >= min_overlap;
        const size_t mem_mb = getenv_mb("BGNS_MEM_MB", 256);
        if (use_blas){
            int Pa=1,Pb=1; dense_panel_sizes(n, px, py, mem_mb, Pa, Pb);
            vector<int> colsA(Pa), colsB(Pb); vector<double> A,B,C,ssA,ssB;
            if (Rf_isReal(_x)){
                const double *X=REAL(_x);
                if (haveY){
                    const double *Y = Rf_isReal(_y) ? REAL(_y) : nullptr;
                    const int *YI = Rf_isReal(_y) ? nullptr : INTEGER(_y);
                    for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                        const int pa=std::min(Pa, px-i0); for(int j=0;j<pa;++j) colsA[j]=i0+j; build_dense_panel_X_real(X,n,colsA.data(),pa,A,ssA);
                        for(int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt();
                            const int pb=std::min(Pb, py-j0); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            if (Rf_isReal(_y)) build_dense_panel_X_real(Y,n,colsB.data(),pb,B,ssB); else build_dense_panel_X_int(YI,n,colsB.data(),pb,B,ssB);
                            C.assign((size_t)pa*(size_t)pb,0.0); dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for(int jb=0;jb<pb;++jb){ const int gj=j0+jb; const double ss_t=ssB[jb]; if(!(ss_t>0.0)) continue;
                                for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; const double ss_s=ssA[ia]; if(!(ss_s>0.0)){ ans[(size_t)gi + (size_t)px*(size_t)gj]=NA_REAL; continue; }
                                    const double num=C[(size_t)ia+(size_t)jb*(size_t)pa]; const double den=std::sqrt(ss_s*ss_t); if(den>0.0) ans[(size_t)gi + (size_t)px*(size_t)gj]=num/den; } }
                        }
                        progress.next();
                    }
                } else {
                    for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                        const int pa=std::min(Pa, px-i0); for(int j=0;j<pa;++j) colsA[j]=i0+j; build_dense_panel_X_real(X,n,colsA.data(),pa,A,ssA);
                        for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; ans[(size_t)gi*px+gi]=(ssA[ia]>0.0?1.0:NA_REAL); }
                        for(int j0=i0;j0<px;j0+=Pb){ bgns_check_interrupt();
                            const int pb=std::min(Pb, px-j0); for(int j=0;j<pb;++j) colsB[j]=j0+j; build_dense_panel_X_real(X,n,colsB.data(),pb,B,ssB);
                            C.assign((size_t)pa*(size_t)pb,0.0); dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for(int jb=0;jb<pb;++jb){ const int gj=j0+jb; const double ss_t=ssB[jb]; if(!(ss_t>0.0)) continue;
                                for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; if(!(gi<gj)) continue; const double ss_s=ssA[ia]; if(!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia+(size_t)jb*(size_t)pa]; const double den=std::sqrt(ss_s*ss_t); if(den>0.0){ double r=num/den; ans[(size_t)gi*px+gj]=r; ans[(size_t)gj*px+gi]=r; } } }
                        }
                        progress.next();
                    }
                }
            } else {
                const int *XI=INTEGER(_x);
                if (haveY){
                    const double *Y = Rf_isReal(_y) ? REAL(_y) : nullptr;
                    const int *YI   = Rf_isReal(_y) ? nullptr   : INTEGER(_y);
                    for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                        const int pa=std::min(Pa, px-i0); for(int j=0;j<pa;++j) colsA[j]=i0+j; build_dense_panel_X_int(XI,n,colsA.data(),pa,A,ssA);
                        for(int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt();
                            const int pb=std::min(Pb, py-j0); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            if (Rf_isReal(_y)) build_dense_panel_X_real(Y,n,colsB.data(),pb,B,ssB); else build_dense_panel_X_int(YI,n,colsB.data(),pb,B,ssB);
                            C.assign((size_t)pa*(size_t)pb,0.0); dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for(int jb=0;jb<pb;++jb){ const int gj=j0+jb; const double ss_t=ssB[jb]; if(!(ss_t>0.0)) continue;
                                for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; const double ss_s=ssA[ia]; if(!(ss_s>0.0)){ ans[(size_t)gi + (size_t)px*(size_t)gj]=NA_REAL; continue; }
                                    const double num=C[(size_t)ia+(size_t)jb*(size_t)pa]; const double den=std::sqrt(ss_s*ss_t); if(den>0.0) ans[(size_t)gi + (size_t)px*(size_t)gj]=num/den; } }
                        }
                        progress.next();
                    }
                } else {
                    for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                        const int pa=std::min(Pa, px-i0); for(int j=0;j<pa;++j) colsA[j]=i0+j; build_dense_panel_X_int(XI,n,colsA.data(),pa,A,ssA);
                        for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; ans[(size_t)gi*px+gi]=(ssA[ia]>0.0?1.0:NA_REAL); }
                        for(int j0=i0;j0<px;j0+=Pb){ bgns_check_interrupt();
                            const int pb=std::min(Pb, px-j0); for(int j=0;j<pb;++j) colsB[j]=j0+j; build_dense_panel_X_int(XI,n,colsB.data(),pb,B,ssB);
                            C.assign((size_t)pa*(size_t)pb,0.0); dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for(int jb=0;jb<pb;++jb){ const int gj=j0+jb; const double ss_t=ssB[jb]; if(!(ss_t>0.0)) continue;
                                for(int ia=0;ia<pa;++ia){ const int gi=i0+ia; if(!(gi<gj)) continue; const double ss_s=ssA[ia]; if(!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia+(size_t)jb*(size_t)pa]; const double den=std::sqrt(ss_s*ss_t); if(den>0.0){ double r=num/den; ans[(size_t)gi*px+gj]=r; ans[(size_t)gj*px+gi]=r; } } }
                        }
                        progress.next();
                    }
                }
            }
        } else {
            // NA-aware FULL/SPARSE panelized path (mixed for dense/int)
            const bool xReal = Rf_isReal(_x); const bool yReal = haveY ? Rf_isReal(_y) : xReal;
            const bool use16 = (n <= 65535); const int Pa = use16 ? sparse_panel_cols<uint16_t>(n, px, mem_mb) : sparse_panel_cols<uint32_t>(n, px, mem_mb);
            const int Pb = use16 ? sparse_panel_cols<uint16_t>(n, (haveY?py:px), mem_mb) : sparse_panel_cols<uint32_t>(n, (haveY?py:px), mem_mb);
            vector<int> colsA(Pa), colsB(Pb);
            auto compute_tile = [&](auto &PX, auto &PY, bool triangular_same_panel){
                const int Pxa=PX.P, Pya=PY.P; [[maybe_unused]] const int O=omp_thresh();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) if (Pya >= O) num_threads(bgns_omp_threads())
#endif
                for(int tj=0;tj<Pya;++tj){
                    if(!PY.ok_local[tj]) continue; const int gj = PY.local_cols[tj];
                    for(int si=0;si<Pxa;++si){
                        if(!PX.ok_local[si]) continue; const int gi = PX.local_cols[si];
                        if(!haveY && triangular_same_panel && !(gi<gj)) continue;
                        double r = NA_REAL;
                        bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                        const int min_overlap = getenv_int("BGNS_MIN_OVERLAP", 3);
                        const bool use_inter_den = Rf_asLogical(_use_inter_den);
                        const bool pairwise_complete = Rf_asLogical(_pairwise);

                        if (pairwise_complete){
                            if (Sd && Td){
                                const double *a=PX.D.mv_col_ptr(PX.map_dense[si]), *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                const double ss_s=PX.D.sumsq[PX.map_dense[si]], ss_t=PY.D.sumsq[PY.map_dense[tj]];
                                if(n>=min_overlap && ss_s>0.0 && ss_t>0.0){ double num=dot_dense_dense_num(a,b,n); double den=std::sqrt(ss_s*ss_t); if(den>0.0) r=num/den; }
                            } else if (Sd && !Td){
                                const int tpos=PY.map_sparse[tj]; const int k=PY.S.nnz_col(tpos); if(k>=min_overlap){
                                    const double *a=PX.D.mv_col_ptr(PX.map_dense[si]);
                                    if(use_inter_den){ auto st=full_sparse_inter(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k, PY.S.sumsq[tpos]); double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den; }
                                    else { auto st=full_sparse_num(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k); double den=std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[tpos]); if(den>0.0) r=st.num/den; } }
                            } else if (!Sd && Td){
                                const int spos=PX.map_sparse[si]; const int k=PX.S.nnz_col(spos); if(k>=min_overlap){
                                    const double *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                    if(use_inter_den){ auto st=full_sparse_inter(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k, PX.S.sumsq[spos]); double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den; }
                                    else { auto st=full_sparse_num(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k); double den=std::sqrt(PX.S.sumsq[spos] * PY.D.sumsq[PY.map_dense[tj]]); if(den>0.0) r=st.num/den; } }
                            } else {
                                const int spos=PX.map_sparse[si], tpos=PY.map_sparse[tj]; int kS=PX.S.nnz_col(spos), kT=PY.S.nnz_col(tpos);
                                if(kS>=min_overlap && kT>=min_overlap){
                                    bool ok_est=true; if(PX.S.has_bits[spos] && PY.S.has_bits[tpos]){ int ov=PX.S.bits[spos].popcnt_and(PY.S.bits[tpos]); ok_est=(ov>=min_overlap); }
                                    if(ok_est){
                                        if(use_inter_den){ auto st=gallop_inter(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), kS, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kT);
                                            if(st.overlap>=min_overlap){ double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den; } }
                                        else { auto st=gallop_num(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), kS, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kT);
                                            if(st.overlap>=min_overlap){ double den=std::sqrt(PX.S.sumsq[spos]*PY.S.sumsq[tpos]); if(den>0.0) r=st.num/den; } }
                                    }
                                }
                            }
                        } else {
                            if (Sd && Td){
                                const double *a=PX.D.mv_col_ptr(PX.map_dense[si]), *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                const double ss_s=PX.D.sumsq[PX.map_dense[si]], ss_t=PY.D.sumsq[PY.map_dense[tj]];
                                if(n>=min_overlap && ss_s>0.0 && ss_t>0.0){ double num=dot_dense_dense_num(a,b,n); double den=std::sqrt(ss_s*ss_t); if(den>0.0) r=num/den; }
                            }
                        }
                        if(R_FINITE(r)){ if(haveY){ ans[(size_t)gi + (size_t)px*(size_t)gj]=r; } else { ans[(size_t)gi*px+gj]=r; ans[(size_t)gj*px+gi]=r; } }
                    }
                }
            };

            auto set_self_diagonal = [&](const auto &PX){
                for(int si=0; si<PX.P; ++si){
                    const int gi = PX.local_cols[si];
                    bool ok = PX.ok_local[si] && n >= min_overlap;
                    if(ok){
                        if(PX.is_dense[si]){
                            const int dpos = PX.map_dense[si];
                            ok = dpos >= 0 && PX.D.sumsq[dpos] > 0.0;
                        } else if(!pairwise_complete){
                            ok = false;
                        } else {
                            const int spos = PX.map_sparse[si];
                            ok = spos >= 0 &&
                                 PX.S.nnz_col(spos) >= min_overlap &&
                                 PX.S.sumsq[spos] > 0.0;
                        }
                    }
                    ans[(size_t)gi*px+gi] = ok ? 1.0 : NA_REAL;
                }
            };

            if(haveY){
                for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                    const int pa=std::min(Pa, px-i0); vector<int> colsA(pa); for(int i=0;i<pa;++i) colsA[i]=i0+i;
                    if((n<=65535)){ PanelMixed<uint16_t> PX; if(xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x),n,colsA.data(),pa,PX); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x),n,colsA.data(),pa,PX);
                        for(int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt(); const int pb=std::min(Pb, py-j0); vector<int> colsB(pb); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            PanelMixed<uint16_t> PY; if(yReal) build_panel_mixed_from_real<uint16_t>(REAL(_y),n,colsB.data(),pb,PY); else build_panel_mixed_from_int<uint16_t>(INTEGER(_y),n,colsB.data(),pb,PY);
                            compute_tile(PX,PY,false); } }
                    else { PanelMixed<uint32_t> PX; if(xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x),n,colsA.data(),pa,PX); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x),n,colsA.data(),pa,PX);
                        for(int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt(); const int pb=std::min(Pb, py-j0); vector<int> colsB(pb); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            PanelMixed<uint32_t> PY; if(yReal) build_panel_mixed_from_real<uint32_t>(REAL(_y),n,colsB.data(),pb,PY); else build_panel_mixed_from_int<uint32_t>(INTEGER(_y),n,colsB.data(),pb,PY);
                            compute_tile(PX,PY,false); } }
                    progress.next();
                }
            } else {
                for(int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                    const int pa=std::min(Pa, px-i0); vector<int> colsA(pa); for(int i=0;i<pa;++i) colsA[i]=i0+i;
                    if((n<=65535)){ PanelMixed<uint16_t> PX; if(xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x),n,colsA.data(),pa,PX); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x),n,colsA.data(),pa,PX);
                        set_self_diagonal(PX);
                        for(int j0=i0;j0<px;j0+=Pb){ bgns_check_interrupt(); const int pb=std::min(Pb, px-j0); vector<int> colsB(pb); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            PanelMixed<uint16_t> PY; if(xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x),n,colsB.data(),pb,PY); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x),n,colsB.data(),pb,PY);
                            compute_tile(PX,PY,(i0==j0)); } }
                    else { PanelMixed<uint32_t> PX; if(xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x),n,colsA.data(),pa,PX); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x),n,colsA.data(),pa,PX);
                        set_self_diagonal(PX);
                        for(int j0=i0;j0<px;j0+=Pb){ bgns_check_interrupt(); const int pb=std::min(Pb, px-j0); vector<int> colsB(pb); for(int j=0;j<pb;++j) colsB[j]=j0+j;
                            PanelMixed<uint32_t> PY; if(xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x),n,colsB.data(),pb,PY); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x),n,colsB.data(),pb,PY);
                            compute_tile(PX,PY,(i0==j0)); } }
                    progress.next();
                }
            }
        }

        SEXP rdim; rprotect(rdim = RSaneAllocVector(INTSXP,2)); INTEGER(rdim)[0]=px; INTEGER(rdim)[1]=py; Rf_setAttrib(answer, R_DimSymbol, rdim);
        SEXP xdn=Rf_getAttrib(_x,R_DimNamesSymbol); SEXP xcols=(Rf_isNull(xdn)||Rf_xlength(xdn)!=2)?R_NilValue:VECTOR_ELT(xdn,1); SEXP ycols=R_NilValue;
        if(haveY){ SEXP ydn=Rf_getAttrib(_y,R_DimNamesSymbol); if(!Rf_isNull(ydn)&&Rf_xlength(ydn)==2) ycols=VECTOR_ELT(ydn,1); }
        const bool have_xcols=!Rf_isNull(xcols), have_ycols=!Rf_isNull(ycols);
        if(have_xcols || (haveY && have_ycols)){ SEXP rdm; rprotect(rdm = RSaneAllocVector(VECSXP,2));
            SET_VECTOR_ELT(rdm,0, have_xcols?xcols:R_NilValue); if(haveY) SET_VECTOR_ELT(rdm,1, have_ycols?ycols:R_NilValue); else SET_VECTOR_ELT(rdm,1, have_xcols?xcols:R_NilValue);
            Rf_setAttrib(answer, R_DimNamesSymbol, rdm);
        } else if (haveY){ SEXP rdm; rprotect(rdm = RSaneAllocVector(VECSXP,2)); SET_VECTOR_ELT(rdm,0,R_NilValue); SET_VECTOR_ELT(rdm,1,R_NilValue); Rf_setAttrib(answer, R_DimNamesSymbol, rdm); }
    } catch(const std::bad_alloc&) { rerror("Out of memory"); }
      catch(const std::exception &e){ rerror("%s", e.what()); }
    rreturn(answer);
}

// ====== Tidy streaming ======
SEXP C_bicor_tidy(SEXP _x, SEXP _y, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, SEXP _envir)
{
    SEXP ans_df = R_NilValue; int __prot_count = 0;
    try {
        TGStat tgstat(_envir);
        const bool haveY = !Rf_isNull(_y);
        const bool pairwise_complete = Rf_asLogical(_pairwise);
        const double threshold = Rf_asReal(_threshold);
        const bool use_inter_den = Rf_asLogical(_use_inter_den);
        const int min_overlap = getenv_int("BGNS_MIN_OVERLAP", 3);

        bool x_is_mat = Rf_isMatrix(_x) && (Rf_isReal(_x) || Rf_isInteger(_x));
        bool x_is_csc = Rf_isS4(_x) && Rf_inherits(_x, "dgCMatrix");
        if (!(x_is_mat || x_is_csc)) verror("\"x\" must be a numeric matrix or dgCMatrix");
        bool y_is_mat=false, y_is_csc=false;
        if (haveY) {
            y_is_mat = Rf_isMatrix(_y) && (Rf_isReal(_y) || Rf_isInteger(_y));
            y_is_csc = Rf_isS4(_y) && Rf_inherits(_y, "dgCMatrix");
            if (!(y_is_mat || y_is_csc)) verror("\"y\" must be a numeric matrix or dgCMatrix");
        }

        int n=0, px=0, py=0;
        if (x_is_mat) { n=Rf_nrows(_x); px=Rf_ncols(_x); }
        else { CSCRef X; load_csc(_x, X); n=X.nrow; px=X.ncol; }
        if (haveY) {
            if (y_is_mat) { if (Rf_nrows(_y)!=n) verror("x and y must have same number of rows"); py=Rf_ncols(_y); }
            else { CSCRef Y; load_csc(_y, Y); if (Y.nrow!=n) verror("x and y must have same number of rows"); py=Y.ncol; }
        } else py=px;

        bool na_free_x = true, na_free_y = true;
        if (x_is_mat) {
            if (Rf_isReal(_x)) {
                const double *xp = REAL(_x);
                for (int j=0;j<px && na_free_x;++j) for (int i=0;i<n;++i) if (!R_FINITE(xp[(size_t)j*n+i])) { na_free_x=false; break; }
            } else {
                const int *xi = INTEGER(_x);
                for (int j=0;j<px && na_free_x;++j) for (int i=0;i<n;++i) if (xi[(size_t)j*n+i]==NA_INTEGER) { na_free_x=false; break; }
            }
        } else { CSCRef X; load_csc(_x,X); na_free_x = csc_all_finite(X); }
        if (haveY) {
            if (y_is_mat) {
                if (Rf_isReal(_y)) {
                    const double *yp = REAL(_y);
                    for (int j=0;j<py && na_free_y;++j) for (int i=0;i<n;++i) if (!R_FINITE(yp[(size_t)j*n+i])) { na_free_y=false; break; }
                } else {
                    const int *yi = INTEGER(_y);
                    for (int j=0;j<py && na_free_y;++j) for (int i=0;i<n;++i) if (yi[(size_t)j*n+i]==NA_INTEGER) { na_free_y=false; break; }
                }
            } else { CSCRef Y; load_csc(_y,Y); na_free_y = csc_all_finite(Y); }
        }
        const bool use_blas = (!use_inter_den) && na_free_x && (!haveY || na_free_y) && n >= min_overlap;

        ProgressReporter progress(TGStat::instance(), haveY ? (uint64_t)py : (uint64_t)px);
        vector<int> out_i, out_j; vector<double> out_r;
        const size_t mem_mb = getenv_mb("BGNS_MEM_MB", 256);

        auto push_pair = [&](int i, int j, double r){
            if (R_FINITE(r) && std::fabs(r) >= threshold) {
                out_i.push_back(i + 1); out_j.push_back(j + 1); out_r.push_back(r);
            }
        };

        if (use_blas) {
            int Pa=1,Pb=1; dense_panel_sizes(n, px, py, mem_mb, Pa, Pb);
            vector<int> colsA(Pa), colsB(Pb); vector<double> A,B,C,ssA,ssB;
            if (haveY) {
                for (int j0=0;j0<py;j0+=Pb) { bgns_check_interrupt();
                    const int pb=min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                    if (y_is_mat) {
                        if (Rf_isReal(_y)) build_dense_panel_X_real(REAL(_y), n, colsB.data(), pb, B, ssB);
                        else               build_dense_panel_X_int (INTEGER(_y), n, colsB.data(), pb, B, ssB);
                    } else {
                        CSCRef Y; load_csc(_y, Y);
                        build_dense_panel_X_csc(Y.p, Y.i, Y.x, n, colsB.data(), pb, B, ssB);
                    }
                    vector<vector<pair<int,double>>> keep(pb);
                    for (int i0=0;i0<px;i0+=Pa) { bgns_check_interrupt();
                        const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                        if (x_is_mat) {
                            if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsA.data(), pa, A, ssA);
                            else               build_dense_panel_X_int (INTEGER(_x), n, colsA.data(), pa, A, ssA);
                        } else {
                            CSCRef X; load_csc(_x, X);
                            build_dense_panel_X_csc(X.p, X.i, X.x, n, colsA.data(), pa, A, ssA);
                        }
                        C.assign((size_t)pa*(size_t)pb, 0.0);
                        dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                        for (int jb=0;jb<pb;++jb) {
                            if (!(ssB[jb]>0.0)) continue;
                            auto &buf = keep[jb];
                            for (int ia=0;ia<pa;++ia) {
                                const double ss_s = ssA[ia]; if (!(ss_s>0.0)) continue;
                                const double num = C[(size_t)ia + (size_t)jb*(size_t)pa];
                                const double den = std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                const double r = num / den;
                                if (R_FINITE(r) && std::fabs(r)>=threshold) buf.emplace_back(i0+ia, r);
                            }
                        }
                    }
                    for (int jb=0;jb<pb;++jb){ const int gj=j0+jb; auto &buf=keep[jb]; for(auto &pr: buf) push_pair(pr.first, gj, pr.second); }
                    progress.next();
                }
            } else {
                for (int i0=0;i0<px;i0+=Pa) { bgns_check_interrupt();
                    const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                    if (x_is_mat) {
                        if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsA.data(), pa, A, ssA);
                        else               build_dense_panel_X_int (INTEGER(_x), n, colsA.data(), pa, A, ssA);
                    } else {
                        CSCRef X; load_csc(_x, X);
                        build_dense_panel_X_csc(X.p, X.i, X.x, n, colsA.data(), pa, A, ssA);
                    }
                    for (int j0=i0;j0<px;j0+=Pb) { bgns_check_interrupt();
                        const int pb=min(Pb, px-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                        if (x_is_mat) {
                            if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsB.data(), pb, B, ssB);
                            else               build_dense_panel_X_int (INTEGER(_x), n, colsB.data(), pb, B, ssB);
                        } else {
                            CSCRef X; load_csc(_x, X);
                            build_dense_panel_X_csc(X.p, X.i, X.x, n, colsB.data(), pb, B, ssB);
                        }
                        C.assign((size_t)pa*(size_t)pb, 0.0);
                        dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                        for (int jb=0;jb<pb;++jb) {
                            const int gj=j0+jb; if (!(ssB[jb]>0.0)) continue;
                            for (int ia=0;ia<pa;++ia) {
                                const int gi=i0+ia; if (!(gi<gj)) continue;
                                const double ss_s = ssA[ia]; if (!(ss_s>0.0)) continue;
                                const double num = C[(size_t)ia + (size_t)jb*(size_t)pa];
                                const double den = std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                const double r = num / den;
                                push_pair(gi, gj, r);
                            }
                        }
                    }
                    progress.next();
                }
                if (!out_i.empty()) {
                    vector<size_t> idx(out_i.size()); std::iota(idx.begin(), idx.end(), (size_t)0);
                    std::stable_sort(idx.begin(), idx.end(),
                        [&](size_t a, size_t b){ return (out_i[a] < out_i[b]) || (out_i[a]==out_i[b] && out_j[a]<out_j[b]); });
                    vector<int> oi, oj; vector<double> orr; oi.reserve(idx.size()); oj.reserve(idx.size()); orr.reserve(idx.size());
                    for (size_t t=0;t<idx.size();++t){ size_t k=idx[t]; oi.push_back(out_i[k]); oj.push_back(out_j[k]); orr.push_back(out_r[k]); }
                    out_i.swap(oi); out_j.swap(oj); out_r.swap(orr);
                }
            }
        } else {
            const bool use16 = (n<=65535);
            const int Pa = use16 ? sparse_panel_cols<uint16_t>(n, px, mem_mb) : sparse_panel_cols<uint32_t>(n, px, mem_mb);
            const int Pb = use16 ? sparse_panel_cols<uint16_t>(n, (haveY?py:px), mem_mb) : sparse_panel_cols<uint32_t>(n, (haveY?py:px), mem_mb);
            vector<int> colsA(Pa), colsB(Pb);
            auto compute_pair = [&](double r, int gi, int gj){
                if (R_FINITE(r) && std::fabs(r) >= threshold) { out_i.push_back(gi+1); out_j.push_back(gj+1); out_r.push_back(r); }
            };
            auto compute_tile = [&](auto &PX, auto &PY, bool triangular_same_panel){
                const int Pxa=PX.P, Pya=PY.P; [[maybe_unused]] const int O=omp_thresh();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) if (Pya >= O) num_threads(bgns_omp_threads())
#endif
                for (int tj=0;tj<Pya;++tj){
                    if (!PY.ok_local[tj]) continue;
                    const int gj = PY.local_cols[tj];
                    vector<pair<int,double>> buf; buf.reserve(128);
                    for (int si=0;si<Pxa;++si){
                        if (!PX.ok_local[si]) continue;
                        const int gi = PX.local_cols[si];
                        if (!haveY && triangular_same_panel && !(gi<gj)) continue;
                        double r = NA_REAL;
                        const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                        if (pairwise_complete){
                            if (Sd && Td){
                                const double *a=PX.D.mv_col_ptr(PX.map_dense[si]), *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                const double ss_s=PX.D.sumsq[PX.map_dense[si]], ss_t=PY.D.sumsq[PY.map_dense[tj]];
                                if (n>=min_overlap && ss_s>0.0 && ss_t>0.0){ double num=dot_dense_dense_num(a,b,n); double den=std::sqrt(ss_s*ss_t); if(den>0.0) r=num/den; }
                            } else if (Sd && !Td){
                                const int tpos=PY.map_sparse[tj]; const int k=PY.S.nnz_col(tpos);
                                if (k>=min_overlap){
                                    const double *a=PX.D.mv_col_ptr(PX.map_dense[si]);
                                    if (use_inter_den){
                                        auto st=full_sparse_inter(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k, PY.S.sumsq[tpos]); double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den;
                                    } else {
                                        auto st=full_sparse_num(a, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), k); double den=std::sqrt(PX.D.sumsq[PX.map_dense[si]]*PY.S.sumsq[tpos]); if(den>0.0) r=st.num/den;
                                    }
                                }
                            } else if (!Sd && Td){
                                const int spos=PX.map_sparse[si]; const int k=PX.S.nnz_col(spos);
                                if (k>=min_overlap){
                                    const double *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                    if (use_inter_den){
                                        auto st=full_sparse_inter(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k, PX.S.sumsq[spos]); double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den;
                                    } else {
                                        auto st=full_sparse_num(b, PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), k); double den=std::sqrt(PX.S.sumsq[spos]*PY.D.sumsq[PY.map_dense[tj]]); if(den>0.0) r=st.num/den;
                                    }
                                }
                            } else {
                                const int spos=PX.map_sparse[si]; const int tpos=PY.map_sparse[tj]; const int kS=PX.S.nnz_col(spos), kT=PY.S.nnz_col(tpos);
                                if (kS>=min_overlap && kT>=min_overlap){
                                    bool ok=true; if (PX.S.has_bits[spos] && PY.S.has_bits[tpos]){ int ov=PX.S.bits[spos].popcnt_and(PY.S.bits[tpos]); ok=(ov>=min_overlap); }
                                    if (ok){
                                        if (use_inter_den){
                                            auto st=gallop_inter(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), kS, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kT);
                                            if (st.overlap>=min_overlap){ double den=std::sqrt(st.sA*st.sB); if(den>0.0) r=st.num/den; }
                                        } else {
                                            auto st=gallop_num(PX.S.idx_ptr(spos), PX.S.mv_ptr(spos), kS, PY.S.idx_ptr(tpos), PY.S.mv_ptr(tpos), kT);
                                            if (st.overlap>=min_overlap){ double den=std::sqrt(PX.S.sumsq[spos]*PY.S.sumsq[tpos]); if(den>0.0) r=st.num/den; }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (Sd && Td){
                                const double *a=PX.D.mv_col_ptr(PX.map_dense[si]), *b=PY.D.mv_col_ptr(PY.map_dense[tj]);
                                const double ss_s=PX.D.sumsq[PX.map_dense[si]], ss_t=PY.D.sumsq[PY.map_dense[tj]];
                                if (n>=min_overlap && ss_s>0.0 && ss_t>0.0){ double num=dot_dense_dense_num(a,b,n); double den=std::sqrt(ss_s*ss_t); if(den>0.0) r=num/den; }
                            }
                        }
                        if (R_FINITE(r) && std::fabs(r)>=threshold) buf.emplace_back(gi, r);
                    }
#ifdef _OPENMP
#pragma omp critical
#endif
                    { for (auto &pr: buf) compute_pair(pr.second, pr.first, gj); }
                }
            };

            if (haveY) {
                const bool use16b = (n<=65535);
                for (int j0=0;j0<py;j0+=Pb) { bgns_check_interrupt();
                    const int pb=min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                    if (use16b) {
                        PanelMixed<uint16_t> PY;
                        if (y_is_mat) {
                            if (Rf_isReal(_y)) build_panel_mixed_from_real<uint16_t>(REAL(_y), n, colsB.data(), pb, PY);
                            else               build_panel_mixed_from_int <uint16_t>(INTEGER(_y), n, colsB.data(), pb, PY);
                        } else { CSCRef Y; load_csc(_y,Y); build_panel_mixed_from_csc<uint16_t>(Y.p,Y.i,Y.x,n, colsB.data(), pb, PY); }
                        for (int i0=0;i0<px;i0+=Pa) { bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            PanelMixed<uint16_t> PX;
                            if (x_is_mat) {
                                if (Rf_isReal(_x)) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsA.data(), pa, PX);
                                else               build_panel_mixed_from_int <uint16_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint16_t>(X.p,X.i,X.x,n, colsA.data(), pa, PX); }
                            compute_tile(PX,PY,false);
                        }
                    } else {
                        PanelMixed<uint32_t> PY;
                        if (y_is_mat) {
                            if (Rf_isReal(_y)) build_panel_mixed_from_real<uint32_t>(REAL(_y), n, colsB.data(), pb, PY);
                            else               build_panel_mixed_from_int <uint32_t>(INTEGER(_y), n, colsB.data(), pb, PY);
                        } else { CSCRef Y; load_csc(_y,Y); build_panel_mixed_from_csc<uint32_t>(Y.p,Y.i,Y.x,n, colsB.data(), pb, PY); }
                        for (int i0=0;i0<px;i0+=Pa) { bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            PanelMixed<uint32_t> PX;
                            if (x_is_mat) {
                                if (Rf_isReal(_x)) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsA.data(), pa, PX);
                                else               build_panel_mixed_from_int <uint32_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint32_t>(X.p,X.i,X.x,n, colsA.data(), pa, PX); }
                            compute_tile(PX,PY,false);
                        }
                    }
                    progress.next();
                }
            } else {
                const bool use16b = (n<=65535);
                for (int j0=0;j0<px;j0+=Pb) { bgns_check_interrupt();
                    const int pb=min(Pb, px-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                    if (use16b) {
                        PanelMixed<uint16_t> PY;
                        if (x_is_mat) {
                            if (Rf_isReal(_x)) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsB.data(), pb, PY);
                            else               build_panel_mixed_from_int <uint16_t>(INTEGER(_x), n, colsB.data(), pb, PY);
                        } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint16_t>(X.p,X.i,X.x,n, colsB.data(), pb, PY); }
                        for (int i0=0;i0<=j0;i0+=Pa) { bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            PanelMixed<uint16_t> PX;
                            if (x_is_mat) {
                                if (Rf_isReal(_x)) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsA.data(), pa, PX);
                                else               build_panel_mixed_from_int <uint16_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint16_t>(X.p,X.i,X.x,n, colsA.data(), pa, PX); }
                            compute_tile(PX,PY,(i0==j0));
                        }
                    } else {
                        PanelMixed<uint32_t> PY;
                        if (x_is_mat) {
                            if (Rf_isReal(_x)) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsB.data(), pb, PY);
                            else               build_panel_mixed_from_int <uint32_t>(INTEGER(_x), n, colsB.data(), pb, PY);
                        } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint32_t>(X.p,X.i,X.x,n, colsB.data(), pb, PY); }
                        for (int i0=0;i0<=j0;i0+=Pa) { bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            PanelMixed<uint32_t> PX;
                            if (x_is_mat) {
                                if (Rf_isReal(_x)) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsA.data(), pa, PX);
                                else               build_panel_mixed_from_int <uint32_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            } else { CSCRef X; load_csc(_x,X); build_panel_mixed_from_csc<uint32_t>(X.p,X.i,X.x,n, colsA.data(), pa, PX); }
                            compute_tile(PX,PY,(i0==j0));
                        }
                    }
                    progress.next();
                }
                if (!out_i.empty()) {
                    vector<size_t> idx(out_i.size()); std::iota(idx.begin(), idx.end(), (size_t)0);
                    std::stable_sort(idx.begin(), idx.end(),
                        [&](size_t a, size_t b){ return (out_i[a] < out_i[b]) || (out_i[a]==out_i[b] && out_j[a]<out_j[b]); });
                    vector<int> oi, oj; vector<double> orr; oi.reserve(idx.size()); oj.reserve(idx.size()); orr.reserve(idx.size());
                    for (size_t t=0;t<idx.size();++t){ size_t k=idx[t]; oi.push_back(out_i[k]); oj.push_back(out_j[k]); orr.push_back(out_r[k]); }
                    out_i.swap(oi); out_j.swap(oj); out_r.swap(orr);
                }
            }
        }

        const R_xlen_t N=(R_xlen_t)out_i.size();
        SEXP rlist, rnames, rlist_names, rcol1, rcol2, rval;
        rprotect(rlist = RSaneAllocVector(VECSXP, 3));
        rprotect(rlist_names = RSaneAllocVector(STRSXP, 3));
        SET_STRING_ELT(rlist_names, 0, Rf_mkChar("col1"));
        SET_STRING_ELT(rlist_names, 1, Rf_mkChar("col2"));
        SET_STRING_ELT(rlist_names, 2, Rf_mkChar("cor"));
        rprotect(rcol1 = RSaneAllocVector(INTSXP, N));
        rprotect(rcol2 = RSaneAllocVector(INTSXP, N));
        rprotect(rval  = RSaneAllocVector(REALSXP, N));
        rprotect(rnames= RSaneAllocVector(INTSXP, N));
        for (R_xlen_t i=0;i<N;++i){ INTEGER(rcol1)[i]=out_i[i]; INTEGER(rcol2)[i]=out_j[i]; REAL(rval)[i]=out_r[i]; INTEGER(rnames)[i]=(int)(i+1); }

        SEXP levX = get_colnames_any(_x, px);
        SEXP levY = haveY ? get_colnames_any(_y, py) : levX;
        if (!Rf_isNull(levX)) { Rf_setAttrib(rcol1, R_LevelsSymbol, levX); Rf_setAttrib(rcol1, R_ClassSymbol, Rf_mkString("factor")); }
        if (!Rf_isNull(levY)) { Rf_setAttrib(rcol2, R_LevelsSymbol, levY); Rf_setAttrib(rcol2, R_ClassSymbol, Rf_mkString("factor")); }

        SET_VECTOR_ELT(rlist, 0, rcol1);
        SET_VECTOR_ELT(rlist, 1, rcol2);
        SET_VECTOR_ELT(rlist, 2, rval);
        Rf_setAttrib(rlist, R_NamesSymbol, rlist_names);
        Rf_setAttrib(rlist, R_ClassSymbol, Rf_mkString("data.frame"));
        Rf_setAttrib(rlist, R_RowNamesSymbol, rnames);
        ans_df = rlist;
    } catch(const std::bad_alloc&) { rerror("Out of memory"); }
      catch(const std::exception &e){ rerror("%s", e.what()); }
    rreturn(ans_df);
}

// ====== Dense-matrix KNN implementation (shared) ======
static SEXP bicor_knn_impl(SEXP _x, SEXP _y, SEXP _knn, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, bool direct_sparse, SEXP _envir)
{
    SEXP ans_df = R_NilValue; int __prot_count = 0;
    try{
        TGStat tgstat(_envir);
        const bool haveY = !Rf_isNull(_y);
        if ((!Rf_isReal(_x) && !Rf_isInteger(_x)) || (haveY && !Rf_isReal(_y) && !Rf_isInteger(_y)))
            verror("\"x\"/\"y\" must be numeric matrices");
        if (!Rf_isMatrix(_x) || (haveY && !Rf_isMatrix(_y)))
            verror("\"x\"/\"y\" must be matrices");

        const int n  = Rf_nrows(_x);
        const int px = Rf_ncols(_x);
        const int py = haveY ? Rf_ncols(_y) : px;
        if (haveY && Rf_nrows(_y) != n) verror("x and y must have same number of rows");

        if (Rf_xlength(_knn) != 1) verror("\"knn\" must be length 1");
        const int requested_k = Rf_asInteger(_knn); if (requested_k < 1) verror("knn must be >= 1");
        const int K = std::min(requested_k, std::max(0, px - (haveY ? 0 : 1)));
        const int pairwise_lgl = Rf_asLogical(_pairwise);
        if (pairwise_lgl == NA_LOGICAL) verror("\"pairwise.complete.obs\" must be TRUE or FALSE");
        const bool pairwise_complete = (pairwise_lgl == TRUE);
        const double threshold = Rf_asReal(_threshold);
        const bool   use_inter_den = Rf_asLogical(_use_inter_den);
        const int    min_overlap = getenv_int("BGNS_MIN_OVERLAP", 3);

        bool na_free_x=true, na_free_y=true;
        if (Rf_isReal(_x)) { const double *xp=REAL(_x); for(int j=0;j<px && na_free_x;++j) for(int i=0;i<n;++i) if(!R_FINITE(xp[(size_t)j*n+i])){ na_free_x=false; break; } }
        else { const int *xi=INTEGER(_x); for(int j=0;j<px && na_free_x;++j) for(int i=0;i<n;++i) if(xi[(size_t)j*n+i]==NA_INTEGER){ na_free_x=false; break; } }
        if (haveY){
            if (Rf_isReal(_y)) { const double *yp=REAL(_y); for(int j=0;j<py && na_free_y;++j) for(int i=0;i<n;++i) if(!R_FINITE(yp[(size_t)j*n+i])){ na_free_y=false; break; } }
            else { const int *yi=INTEGER(_y); for(int j=0;j<py && na_free_y;++j) for(int i=0;i<n;++i) if(yi[(size_t)j*n+i]==NA_INTEGER){ na_free_y=false; break; } }
        }
        const bool use_blas = (!use_inter_den) && na_free_x && (!haveY || na_free_y) && n >= min_overlap;

        ProgressReporter progress(TGStat::instance(), haveY ? (uint64_t)py : (uint64_t)px);
        vector<int> col1, col2, rankv; vector<double> val;
        const size_t mem_mb = getenv_mb("BGNS_MEM_MB", 256);

        auto keep_topK = [&](vector<pair<int,double>> &buf){
            shrink_topk_buffer(buf, K);
        };

        if (use_blas) {
            if (!direct_sparse) {
                const double S = (double)mem_mb * 1024.0 * 1024.0;
                const double need =
                    8.0 * (double)n * (double)px +  // A
                    8.0 * (double)n * (double)py +  // B
                    8.0 * (double)px * (double)py;  // C
                const bool can_full = (need <= S);

                if (can_full) {
                    vector<double> A,B,C,ssX,ssY;
                    {
                        vector<int> cols(px); std::iota(cols.begin(), cols.end(), 0);
                        if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, cols.data(), px, A, ssX);
                        else               build_dense_panel_X_int (INTEGER(_x), n, cols.data(), px, A, ssX);
                    }
                    if (haveY) {
                        vector<int> cols(py); std::iota(cols.begin(), cols.end(), 0);
                        if (Rf_isReal(_y)) build_dense_panel_X_real(REAL(_y), n, cols.data(), py, B, ssY);
                        else               build_dense_panel_X_int (INTEGER(_y), n, cols.data(), py, B, ssY);
                    } else { B=A; ssY=ssX; }

                    C.assign((size_t)px*(size_t)py, 0.0);
                    dgemm_AB('T','N', px, py, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), px);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((haveY?py:px) >= omp_thresh()) num_threads(bgns_omp_threads())
#endif
                    for (int j=0;j<(haveY?py:px);++j){
                        if (!(ssY[j]>0.0)) { progress.next(); continue; }
                        vector<pair<int,double>> buf; buf.reserve(px);
                        for (int i=0;i<px;++i){
                            if (!haveY && i==j) continue;
                            const double ss_s=ssX[i]; if (!(ss_s>0.0)) continue;
                            const double num=C[(size_t)i + (size_t)j*(size_t)px];
                            const double den=std::sqrt(ss_s * ssY[j]); if (!(den>0.0)) continue;
                            const double r = num / den;
                            tiny_topk_consider(buf, K, i, r, threshold);
                        }
                        if (!buf.empty()){
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
#ifdef _OPENMP
#pragma omp critical
#endif
                            {
                                for (int k=0;k<keep;++k){ col1.push_back(j+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                            }
                        }
                        progress.next();
                    }
                } else { direct_sparse = true; }
            }

            if (direct_sparse) {
                int Pa=1,Pb=1; dense_panel_sizes(n, px, (haveY?py:px), mem_mb, Pa, Pb);
                vector<int> colsA(Pa), colsB(Pb); vector<double> A,B,C,ssA,ssB;
                if (haveY) {
                    for (int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt();
                        const int pb=min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                        if (Rf_isReal(_y)) build_dense_panel_X_real(REAL(_y), n, colsB.data(), pb, B, ssB);
                        else               build_dense_panel_X_int (INTEGER(_y), n, colsB.data(), pb, B, ssB);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsA.data(), pa, A, ssA);
                            else               build_dense_panel_X_int (INTEGER(_x), n, colsA.data(), pa, A, ssA);
                            C.assign((size_t)pa*(size_t)pb, 0.0);
                            dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for (int jb=0;jb<pb;++jb){
                                if (!(ssB[jb]>0.0)) continue;
                                auto &buf=knn[jb];
                                for (int ia=0;ia<pa;++ia){
                                    const int src=i0+ia; const double ss_s=ssA[ia]; if (!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia + (size_t)jb*(size_t)pa];
                                    const double den=std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                    const double r = num / den; tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                if ((int)knn[jb].size()>4*K) keep_topK(knn[jb]);
                            }
                        }
                        for (int jb=0;jb<pb;++jb){
                            auto &buf=knn[jb]; if (buf.empty()) continue;
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0;k<keep;++k){ col1.push_back(j0+jb+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                        progress.next();
                    }
                } else {
                    for (int j0=0;j0<px;j0+=Pb){ bgns_check_interrupt();
                        const int pb=min(Pb, px-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                        if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsB.data(), pb, B, ssB);
                        else               build_dense_panel_X_int (INTEGER(_x), n, colsB.data(), pb, B, ssB);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsA.data(), pa, A, ssA);
                            else               build_dense_panel_X_int (INTEGER(_x), n, colsA.data(), pa, A, ssA);
                            C.assign((size_t)pa*(size_t)pb, 0.0);
                            dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for (int jb=0;jb<pb;++jb){
                                const int tgt=j0+jb; if (!(ssB[jb]>0.0)) continue; auto &buf=knn[jb];
                                for (int ia=0;ia<pa;++ia){
                                    const int src=i0+ia; if (src==tgt) continue; const double ss_s=ssA[ia]; if (!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia + (size_t)jb*(size_t)pa];
                                    const double den=std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                    const double r = num / den; tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                if ((int)knn[jb].size()>4*K) keep_topK(knn[jb]);
                            }
                        }
                        for (int jb=0;jb<pb;++jb){
                            auto &buf=knn[jb]; if (buf.empty()) continue;
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0;k<keep;++k){ col1.push_back(j0+jb+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                        progress.next();
                    }
                }
            }
        } else {
            // NA-aware FULL/SPARSE panelized path for dense matrices (mixed for real/int)
            const bool xReal = Rf_isReal(_x);
            const bool yReal = haveY ? Rf_isReal(_y) : xReal;
            const bool use16 = (n <= 65535);
            const int Pa = use16 ? sparse_panel_cols<uint16_t>(n, px, mem_mb) : sparse_panel_cols<uint32_t>(n, px, mem_mb);
            const int Pb = use16 ? sparse_panel_cols<uint16_t>(n, (haveY?py:px), mem_mb) : sparse_panel_cols<uint32_t>(n, (haveY?py:px), mem_mb);
            vector<int> colsA(Pa), colsB(Pb);

            if (haveY) {
                for (int j0=0; j0<py; j0+=Pb) { bgns_check_interrupt();
                    const int pb = std::min(Pb, py - j0);
                    for (int j=0; j<pb; ++j) colsB[j] = j0 + j;
                    if (use16) {
                        PanelMixed<uint16_t> PY; if (yReal) build_panel_mixed_from_real<uint16_t>(REAL(_y), n, colsB.data(), pb, PY); else build_panel_mixed_from_int<uint16_t>(INTEGER(_y), n, colsB.data(), pb, PY);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                            const int pa = std::min(Pa, px - i0);
                            for (int i=0; i<pa; ++i) colsA[i] = i0 + i;
                            PanelMixed<uint16_t> PX; if (xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsA.data(), pa, PX); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            for (int tj=0; tj<PY.P; ++tj) {
                                if (!PY.ok_local[tj]) continue;
                                auto &buf = knn[tj];
                                for (int si=0; si<PX.P; ++si) {
                                    if (!PX.ok_local[si]) continue;
                                    const int src = PX.local_cols[si];
                                    double r = NA_REAL;
                                    const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                                    if (!pairwise_complete && !(Sd && Td)) {
                                        // pairwise.complete.obs = FALSE: only complete columns are eligible
                                    } else if (Sd && Td) {
                                        const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                        const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                        const double ss_s = PX.D.sumsq[PX.map_dense[si]], ss_t = PY.D.sumsq[PY.map_dense[tj]];
                                        if (n>=min_overlap && ss_s>0.0 && ss_t>0.0) { double num = dot_dense_dense_num(a,b,n); double den = std::sqrt(ss_s*ss_t); if (den>0.0) r = num/den; }
                                    } else if (Sd && !Td) {
                                        const int k = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (k>=min_overlap) {
                                            const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                            if (use_inter_den) { auto st = full_sparse_inter(a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k, PY.S.sumsq[PY.map_sparse[tj]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k); double den = std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else if (!Sd && Td) {
                                        const int k = PX.S.nnz_col(PX.map_sparse[si]);
                                        if (k>=min_overlap) {
                                            const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                            if (use_inter_den) { auto st = full_sparse_inter(b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k, PX.S.sumsq[PX.map_sparse[si]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k); double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.D.sumsq[PY.map_dense[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else {
                                        const int ks = PX.S.nnz_col(PX.map_sparse[si]);
                                        const int kt = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (ks>=min_overlap && kt>=min_overlap) {
                                            bool ok = true;
                                            if (PX.S.has_bits[PX.map_sparse[si]] && PY.S.has_bits[PY.map_sparse[tj]]) {
                                                int ov = PX.S.bits[PX.map_sparse[si]].popcnt_and(PY.S.bits[PY.map_sparse[tj]]);
                                                ok = (ov>=min_overlap);
                                            }
                                            if (ok) {
                                                if (use_inter_den) { auto st = gallop_inter(PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; } }
                                                else               { auto st = gallop_num  (PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; } }
                                            }
                                        }
                                    }
                                    tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                keep_topK(buf);
                            }
                        }
                        for (int tj=0; tj<PY.P; ++tj) {
                            auto &buf = knn[tj]; if (buf.empty()) continue;
                            const int keep = (int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(), [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep, [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0; k<keep; ++k) { col1.push_back(j0+tj+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                    } else {
                        PanelMixed<uint32_t> PY; if (yReal) build_panel_mixed_from_real<uint32_t>(REAL(_y), n, colsB.data(), pb, PY); else build_panel_mixed_from_int<uint32_t>(INTEGER(_y), n, colsB.data(), pb, PY);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                            const int pa = std::min(Pa, px - i0);
                            for (int i=0; i<pa; ++i) colsA[i] = i0 + i;
                            PanelMixed<uint32_t> PX; if (xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsA.data(), pa, PX); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            for (int tj=0; tj<PY.P; ++tj) {
                                if (!PY.ok_local[tj]) continue;
                                auto &buf = knn[tj];
                                for (int si=0; si<PX.P; ++si) {
                                    if (!PX.ok_local[si]) continue;
                                    const int src = PX.local_cols[si];
                                    double r = NA_REAL;
                                    const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                                    if (!pairwise_complete && !(Sd && Td)) {
                                        // pairwise.complete.obs = FALSE: only complete columns are eligible
                                    } else if (Sd && Td) {
                                        const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                        const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                        const double ss_s = PX.D.sumsq[PX.map_dense[si]], ss_t = PY.D.sumsq[PY.map_dense[tj]];
                                        if (n>=min_overlap && ss_s>0.0 && ss_t>0.0) { double num = dot_dense_dense_num(a,b,n); double den = std::sqrt(ss_s*ss_t); if (den>0.0) r = num/den; }
                                    } else if (Sd && !Td) {
                                        const int k = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (k>=min_overlap) {
                                            const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                            if (use_inter_den) { auto st = full_sparse_inter(a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k, PY.S.sumsq[PY.map_sparse[tj]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k); double den = std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else if (!Sd && Td) {
                                        const int k = PX.S.nnz_col(PX.map_sparse[si]);
                                        if (k>=min_overlap) {
                                            const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                            if (use_inter_den) { auto st = full_sparse_inter(b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k, PX.S.sumsq[PX.map_sparse[si]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k); double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.D.sumsq[PY.map_dense[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else {
                                        const int ks = PX.S.nnz_col(PX.map_sparse[si]);
                                        const int kt = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (ks>=min_overlap && kt>=min_overlap) {
                                            bool ok = true;
                                            if (PX.S.has_bits[PX.map_sparse[si]] && PY.S.has_bits[PY.map_sparse[tj]]) {
                                                int ov = PX.S.bits[PX.map_sparse[si]].popcnt_and(PY.S.bits[PY.map_sparse[tj]]);
                                                ok = (ov>=min_overlap);
                                            }
                                            if (ok) {
                                                if (use_inter_den) { auto st = gallop_inter(PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; } }
                                                else               { auto st = gallop_num  (PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; } }
                                            }
                                        }
                                    }
                                    tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                keep_topK(buf);
                            }
                        }
                        for (int tj=0; tj<PY.P; ++tj) {
                            auto &buf = knn[tj]; if (buf.empty()) continue;
                            const int keep = (int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(), [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep, [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0; k<keep; ++k) { col1.push_back(j0+tj+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                    }
                }
            } else {
                for (int j0=0; j0<px; j0+=Pb) { bgns_check_interrupt();
                    const int pb = std::min(Pb, px - j0);
                    for (int j=0; j<pb; ++j) colsB[j] = j0 + j;
                    if (use16) {
                        PanelMixed<uint16_t> PY; if (xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsB.data(), pb, PY); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x), n, colsB.data(), pb, PY);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                            const int pa = std::min(Pa, px - i0);
                            for (int i=0; i<pa; ++i) colsA[i] = i0 + i;
                            PanelMixed<uint16_t> PX; if (xReal) build_panel_mixed_from_real<uint16_t>(REAL(_x), n, colsA.data(), pa, PX); else build_panel_mixed_from_int<uint16_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            for (int tj=0; tj<PY.P; ++tj) {
                                if (!PY.ok_local[tj]) continue; const int tgt = PY.local_cols[tj];
                                auto &buf = knn[tj];
                                for (int si=0; si<PX.P; ++si) {
                                    if (!PX.ok_local[si]) continue; const int src = PX.local_cols[si]; if (src==tgt) continue;
                                    double r = NA_REAL;
                                    const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                                    if (!pairwise_complete && !(Sd && Td)) {
                                        // pairwise.complete.obs = FALSE: only complete columns are eligible
                                    } else if (Sd && Td) {
                                        const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                        const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                        const double ss_s = PX.D.sumsq[PX.map_dense[si]], ss_t = PY.D.sumsq[PY.map_dense[tj]];
                                        if (n>=min_overlap && ss_s>0.0 && ss_t>0.0) { double num = dot_dense_dense_num(a,b,n); double den = std::sqrt(ss_s*ss_t); if (den>0.0) r = num/den; }
                                    } else if (Sd && !Td) {
                                        const int k = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (k>=min_overlap) {
                                            const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                            if (use_inter_den) { auto st = full_sparse_inter(a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k, PY.S.sumsq[PY.map_sparse[tj]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k); double den = std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else if (!Sd && Td) {
                                        const int k = PX.S.nnz_col(PX.map_sparse[si]);
                                        if (k>=min_overlap) {
                                            const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                            if (use_inter_den) { auto st = full_sparse_inter(b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k, PX.S.sumsq[PX.map_sparse[si]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k); double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.D.sumsq[PY.map_dense[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else {
                                        const int ks = PX.S.nnz_col(PX.map_sparse[si]);
                                        const int kt = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (ks>=min_overlap && kt>=min_overlap) {
                                            bool ok = true;
                                            if (PX.S.has_bits[PX.map_sparse[si]] && PY.S.has_bits[PY.map_sparse[tj]]) {
                                                int ov = PX.S.bits[PX.map_sparse[si]].popcnt_and(PY.S.bits[PY.map_sparse[tj]]);
                                                ok = (ov>=min_overlap);
                                            }
                                            if (ok) {
                                                if (use_inter_den) { auto st = gallop_inter(PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; } }
                                                else               { auto st = gallop_num  (PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; } }
                                            }
                                        }
                                    }
                                    tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                keep_topK(buf);
                            }
                        }
                        for (int tj=0; tj<PY.P; ++tj) {
                            auto &buf = knn[tj]; if (buf.empty()) continue;
                            const int keep = (int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(), [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep, [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0; k<keep; ++k) { col1.push_back(j0+tj+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                    } else {
                        PanelMixed<uint32_t> PY; if (xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsB.data(), pb, PY); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x), n, colsB.data(), pb, PY);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                            const int pa = std::min(Pa, px - i0);
                            for (int i=0; i<pa; ++i) colsA[i] = i0 + i;
                            PanelMixed<uint32_t> PX; if (xReal) build_panel_mixed_from_real<uint32_t>(REAL(_x), n, colsA.data(), pa, PX); else build_panel_mixed_from_int<uint32_t>(INTEGER(_x), n, colsA.data(), pa, PX);
                            for (int tj=0; tj<PY.P; ++tj) {
                                if (!PY.ok_local[tj]) continue; const int tgt = PY.local_cols[tj];
                                auto &buf = knn[tj];
                                for (int si=0; si<PX.P; ++si) {
                                    if (!PX.ok_local[si]) continue; const int src = PX.local_cols[si]; if (src==tgt) continue;
                                    double r = NA_REAL;
                                    const bool Sd = PX.is_dense[si], Td = PY.is_dense[tj];
                                    if (!pairwise_complete && !(Sd && Td)) {
                                        // pairwise.complete.obs = FALSE: only complete columns are eligible
                                    } else if (Sd && Td) {
                                        const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                        const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                        const double ss_s = PX.D.sumsq[PX.map_dense[si]], ss_t = PY.D.sumsq[PY.map_dense[tj]];
                                        if (n>=min_overlap && ss_s>0.0 && ss_t>0.0) { double num = dot_dense_dense_num(a,b,n); double den = std::sqrt(ss_s*ss_t); if (den>0.0) r = num/den; }
                                    } else if (Sd && !Td) {
                                        const int k = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (k>=min_overlap) {
                                            const double *a = PX.D.mv_col_ptr(PX.map_dense[si]);
                                            if (use_inter_den) { auto st = full_sparse_inter(a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k, PY.S.sumsq[PY.map_sparse[tj]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (a, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), k); double den = std::sqrt(PX.D.sumsq[PX.map_dense[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else if (!Sd && Td) {
                                        const int k = PX.S.nnz_col(PX.map_sparse[si]);
                                        if (k>=min_overlap) {
                                            const double *b = PY.D.mv_col_ptr(PY.map_dense[tj]);
                                            if (use_inter_den) { auto st = full_sparse_inter(b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k, PX.S.sumsq[PX.map_sparse[si]]); double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; }
                                            else               { auto st = full_sparse_num  (b, PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), k); double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.D.sumsq[PY.map_dense[tj]]); if (den>0.0) r = st.num/den; }
                                        }
                                    } else {
                                        const int ks = PX.S.nnz_col(PX.map_sparse[si]);
                                        const int kt = PY.S.nnz_col(PY.map_sparse[tj]);
                                        if (ks>=min_overlap && kt>=min_overlap) {
                                            bool ok = true;
                                            if (PX.S.has_bits[PX.map_sparse[si]] && PY.S.has_bits[PY.map_sparse[tj]]) {
                                                int ov = PX.S.bits[PX.map_sparse[si]].popcnt_and(PY.S.bits[PY.map_sparse[tj]]);
                                                ok = (ov>=min_overlap);
                                            }
                                            if (ok) {
                                                if (use_inter_den) { auto st = gallop_inter(PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(st.sA*st.sB); if (den>0.0) r = st.num/den; } }
                                                else               { auto st = gallop_num  (PX.S.idx_ptr(PX.map_sparse[si]), PX.S.mv_ptr(PX.map_sparse[si]), ks, PY.S.idx_ptr(PY.map_sparse[tj]), PY.S.mv_ptr(PY.map_sparse[tj]), kt); if (st.overlap>=min_overlap) { double den = std::sqrt(PX.S.sumsq[PX.map_sparse[si]] * PY.S.sumsq[PY.map_sparse[tj]]); if (den>0.0) r = st.num/den; } }
                                            }
                                        }
                                    }
                                    tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                keep_topK(buf);
                            }
                        }
                        for (int tj=0; tj<PY.P; ++tj) {
                            auto &buf = knn[tj]; if (buf.empty()) continue;
                            const int keep = (int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(), [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep, [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0; k<keep; ++k) { col1.push_back(j0+tj+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                    }
                }
            }
        }

        const R_xlen_t N = (R_xlen_t)col1.size();
        SEXP rlist, rnames, rlist_names, rcol1, rcol2, rval, rrank;
        rprotect(rlist = RSaneAllocVector(VECSXP, 4));
        rprotect(rlist_names = RSaneAllocVector(STRSXP, 4));
        SET_STRING_ELT(rlist_names, 0, Rf_mkChar("col1"));
        SET_STRING_ELT(rlist_names, 1, Rf_mkChar("col2"));
        SET_STRING_ELT(rlist_names, 2, Rf_mkChar("val"));
        SET_STRING_ELT(rlist_names, 3, Rf_mkChar("rank"));
        rprotect(rcol1 = RSaneAllocVector(INTSXP, N));
        rprotect(rcol2 = RSaneAllocVector(INTSXP, N));
        rprotect(rval  = RSaneAllocVector(REALSXP, N));
        rprotect(rrank = RSaneAllocVector(INTSXP, N));
        rprotect(rnames= RSaneAllocVector(INTSXP, N));
        for (R_xlen_t i=0;i<N;++i){
            INTEGER(rcol1)[i]=col1[i]; INTEGER(rcol2)[i]=col2[i]; REAL(rval)[i]=val[i]; INTEGER(rrank)[i]=rankv[i]; INTEGER(rnames)[i]=(int)(i+1);
        }
        set_knn_factor_levels(rcol1, rcol2, _x, px, _y, py, haveY);
        SET_VECTOR_ELT(rlist, 0, rcol1); SET_VECTOR_ELT(rlist, 1, rcol2); SET_VECTOR_ELT(rlist, 2, rval); SET_VECTOR_ELT(rlist, 3, rrank);
        Rf_setAttrib(rlist, R_NamesSymbol, rlist_names); Rf_setAttrib(rlist, R_ClassSymbol, Rf_mkString("data.frame")); Rf_setAttrib(rlist, R_RowNamesSymbol, rnames);
        ans_df = rlist;
    } catch(const std::bad_alloc&) { rerror("Out of memory"); }
      catch(const std::exception &e){ rerror("%s", e.what()); }
    rreturn(ans_df);
}


// ====== Mixed dense / dgCMatrix KNN ======
SEXP C_bicor_knn_spdem(SEXP _x, SEXP _y, SEXP _knn, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, SEXP _direct_sparse, SEXP _envir)
{
    SEXP ans_df = R_NilValue; int __prot_count = 0;
    try {
        TGStat tgstat(_envir);
        const bool haveY = !Rf_isNull(_y);
        if (!haveY) verror("bicor_knn spdem route requires non-NULL y");

        const bool x_is_mat = Rf_isMatrix(_x) && (Rf_isReal(_x) || Rf_isInteger(_x));
        const bool y_is_mat = Rf_isMatrix(_y) && (Rf_isReal(_y) || Rf_isInteger(_y));
        const bool x_is_csc = Rf_isS4(_x) && Rf_inherits(_x, "dgCMatrix");
        const bool y_is_csc = Rf_isS4(_y) && Rf_inherits(_y, "dgCMatrix");
        if (!(x_is_mat || x_is_csc)) verror("\"x\" must be a numeric matrix or dgCMatrix");
        if (!(y_is_mat || y_is_csc)) verror("\"y\" must be a numeric matrix or dgCMatrix");
        if (x_is_csc == y_is_csc) verror("bicor_knn spdem route requires exactly one dense matrix and one dgCMatrix");

        CSCRef Xc, Yc;
        int n = 0, px = 0, py = 0;
        if (x_is_mat) { n = Rf_nrows(_x); px = Rf_ncols(_x); }
        else { load_csc(_x, Xc); n = Xc.nrow; px = Xc.ncol; }
        if (y_is_mat) { if (Rf_nrows(_y) != n) verror("x and y must have same number of rows"); py = Rf_ncols(_y); }
        else { load_csc(_y, Yc); if (Yc.nrow != n) verror("x and y must have same number of rows"); py = Yc.ncol; }

        if (Rf_xlength(_knn) != 1) verror("\"knn\" must be length 1");
        const int requested_k = Rf_asInteger(_knn); if (requested_k < 1) verror("knn must be >= 1");
        const int K = std::min(requested_k, std::max(0, px - (haveY ? 0 : 1)));
        const int pairwise_lgl = Rf_asLogical(_pairwise);
        if (pairwise_lgl == NA_LOGICAL) verror("\"pairwise.complete.obs\" must be TRUE or FALSE");
        const bool pairwise_complete = (pairwise_lgl == TRUE);
        const double threshold = Rf_asReal(_threshold);
        const bool use_inter_den = Rf_asLogical(_use_inter_den);
        const int direct_sparse_lgl = Rf_asLogical(_direct_sparse);
        if (direct_sparse_lgl == NA_LOGICAL) verror("\"direct_sparse\" must be TRUE or FALSE");
        const bool direct_sparse = (direct_sparse_lgl == TRUE);
        (void)direct_sparse; // mixed input always streams top-k buffers instead of materializing full dense output
        const int min_overlap = getenv_int("BGNS_MIN_OVERLAP", 3);

        bool na_free_x = true, na_free_y = true;
        if (x_is_mat) {
            if (Rf_isReal(_x)) { const double *xp = REAL(_x); for (int j=0;j<px && na_free_x;++j) for (int i=0;i<n;++i) if (!R_FINITE(xp[(size_t)j*n+i])) { na_free_x=false; break; } }
            else { const int *xi = INTEGER(_x); for (int j=0;j<px && na_free_x;++j) for (int i=0;i<n;++i) if (xi[(size_t)j*n+i]==NA_INTEGER) { na_free_x=false; break; } }
        } else na_free_x = csc_all_finite(Xc);
        if (y_is_mat) {
            if (Rf_isReal(_y)) { const double *yp = REAL(_y); for (int j=0;j<py && na_free_y;++j) for (int i=0;i<n;++i) if (!R_FINITE(yp[(size_t)j*n+i])) { na_free_y=false; break; } }
            else { const int *yi = INTEGER(_y); for (int j=0;j<py && na_free_y;++j) for (int i=0;i<n;++i) if (yi[(size_t)j*n+i]==NA_INTEGER) { na_free_y=false; break; } }
        } else na_free_y = csc_all_finite(Yc);
        const bool use_blas = (!use_inter_den) && na_free_x && na_free_y && pairwise_complete && n >= min_overlap;

        vector<int> col1, col2, rankv; vector<double> val;
        const size_t mem_mb = getenv_mb("BGNS_MEM_MB", 256);
        ProgressReporter progress(TGStat::instance(), (uint64_t)py);

        if (use_blas) {
            int Pa=1, Pb=1; dense_panel_sizes(n, px, py, mem_mb, Pa, Pb);
            vector<int> colsA(Pa), colsB(Pb); vector<double> A, B, C, ssA, ssB;
            for (int j0=0; j0<py; j0+=Pb) { bgns_check_interrupt();
                const int pb = min(Pb, py-j0);
                for (int j=0; j<pb; ++j) colsB[j] = j0 + j;
                if (y_is_mat) {
                    if (Rf_isReal(_y)) build_dense_panel_X_real(REAL(_y), n, colsB.data(), pb, B, ssB);
                    else               build_dense_panel_X_int (INTEGER(_y), n, colsB.data(), pb, B, ssB);
                } else {
                    build_dense_panel_X_csc(Yc.p, Yc.i, Yc.x, n, colsB.data(), pb, B, ssB);
                }
                vector<vector<pair<int,double>>> knn(pb);
                for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                    const int pa = min(Pa, px-i0);
                    for (int i=0; i<pa; ++i) colsA[i] = i0 + i;
                    if (x_is_mat) {
                        if (Rf_isReal(_x)) build_dense_panel_X_real(REAL(_x), n, colsA.data(), pa, A, ssA);
                        else               build_dense_panel_X_int (INTEGER(_x), n, colsA.data(), pa, A, ssA);
                    } else {
                        build_dense_panel_X_csc(Xc.p, Xc.i, Xc.x, n, colsA.data(), pa, A, ssA);
                    }
                    C.assign((size_t)pa*(size_t)pb, 0.0);
                    dgemm_AB('T','N', pa, pb, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), pa);
                    for (int jb=0; jb<pb; ++jb) {
                        if (!(ssB[jb] > 0.0)) continue;
                        auto &buf = knn[jb];
                        for (int ia=0; ia<pa; ++ia) {
                            const double ss_s = ssA[ia]; if (!(ss_s > 0.0)) continue;
                            if (n < min_overlap) continue;
                            const double num = C[(size_t)ia + (size_t)jb*(size_t)pa];
                            const double den = std::sqrt(ss_s * ssB[jb]); if (!(den > 0.0)) continue;
                            const double r = num / den;
                            tiny_topk_consider(buf, K, i0 + ia, r, threshold);
                        }
                        shrink_topk_buffer(buf, K);
                    }
                }
                for (int jb=0; jb<pb; ++jb) append_final_topk(knn[jb], j0+jb, K, col1, col2, val, rankv);
                progress.next();
            }
        } else {
            const bool use16 = (n <= 65535);
            const int Pa = use16 ? sparse_panel_cols<uint16_t>(n, px, mem_mb) : sparse_panel_cols<uint32_t>(n, px, mem_mb);
            const int Pb = use16 ? sparse_panel_cols<uint16_t>(n, py, mem_mb) : sparse_panel_cols<uint32_t>(n, py, mem_mb);
            vector<int> colsA(Pa), colsB(Pb);
            if (use16) {
                for (int j0=0; j0<py; j0+=Pb) { bgns_check_interrupt();
                    const int pb = min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                    PanelMixed<uint16_t> PY; build_panel_mixed_any<uint16_t>(_y, y_is_mat, y_is_csc ? &Yc : nullptr, n, colsB.data(), pb, PY);
                    vector<vector<pair<int,double>>> knn(pb);
                    for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                        const int pa = min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                        PanelMixed<uint16_t> PX; build_panel_mixed_any<uint16_t>(_x, x_is_mat, x_is_csc ? &Xc : nullptr, n, colsA.data(), pa, PX);
                        for (int tj=0; tj<PY.P; ++tj) {
                            if (!PY.ok_local[tj]) continue;
                            auto &buf = knn[tj];
                            for (int si=0; si<PX.P; ++si) {
                                if (!PX.ok_local[si]) continue;
                                const double r = panel_mixed_bicor_pair<uint16_t>(PX, si, PY, tj, pairwise_complete, use_inter_den, min_overlap);
                                tiny_topk_consider(buf, K, PX.local_cols[si], r, threshold);
                            }
                            shrink_topk_buffer(buf, K);
                        }
                    }
                    for (int tj=0; tj<PY.P; ++tj) append_final_topk(knn[tj], PY.local_cols[tj], K, col1, col2, val, rankv);
                    progress.next();
                }
            } else {
                for (int j0=0; j0<py; j0+=Pb) { bgns_check_interrupt();
                    const int pb = min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                    PanelMixed<uint32_t> PY; build_panel_mixed_any<uint32_t>(_y, y_is_mat, y_is_csc ? &Yc : nullptr, n, colsB.data(), pb, PY);
                    vector<vector<pair<int,double>>> knn(pb);
                    for (int i0=0; i0<px; i0+=Pa) { bgns_check_interrupt();
                        const int pa = min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                        PanelMixed<uint32_t> PX; build_panel_mixed_any<uint32_t>(_x, x_is_mat, x_is_csc ? &Xc : nullptr, n, colsA.data(), pa, PX);
                        for (int tj=0; tj<PY.P; ++tj) {
                            if (!PY.ok_local[tj]) continue;
                            auto &buf = knn[tj];
                            for (int si=0; si<PX.P; ++si) {
                                if (!PX.ok_local[si]) continue;
                                const double r = panel_mixed_bicor_pair<uint32_t>(PX, si, PY, tj, pairwise_complete, use_inter_den, min_overlap);
                                tiny_topk_consider(buf, K, PX.local_cols[si], r, threshold);
                            }
                            shrink_topk_buffer(buf, K);
                        }
                    }
                    for (int tj=0; tj<PY.P; ++tj) append_final_topk(knn[tj], PY.local_cols[tj], K, col1, col2, val, rankv);
                    progress.next();
                }
            }
        }

        const R_xlen_t N = (R_xlen_t)col1.size();
        SEXP rlist, rnames, rlist_names, rcol1, rcol2, rval, rrank;
        rprotect(rlist = RSaneAllocVector(VECSXP, 4));
        rprotect(rlist_names = RSaneAllocVector(STRSXP, 4));
        SET_STRING_ELT(rlist_names, 0, Rf_mkChar("col1"));
        SET_STRING_ELT(rlist_names, 1, Rf_mkChar("col2"));
        SET_STRING_ELT(rlist_names, 2, Rf_mkChar("val"));
        SET_STRING_ELT(rlist_names, 3, Rf_mkChar("rank"));
        rprotect(rcol1 = RSaneAllocVector(INTSXP, N));
        rprotect(rcol2 = RSaneAllocVector(INTSXP, N));
        rprotect(rval  = RSaneAllocVector(REALSXP, N));
        rprotect(rrank = RSaneAllocVector(INTSXP, N));
        rprotect(rnames= RSaneAllocVector(INTSXP, N));
        for (R_xlen_t i=0;i<N;++i){ INTEGER(rcol1)[i]=col1[i]; INTEGER(rcol2)[i]=col2[i]; REAL(rval)[i]=val[i]; INTEGER(rrank)[i]=rankv[i]; INTEGER(rnames)[i]=(int)(i+1); }
        set_knn_factor_levels(rcol1, rcol2, _x, px, _y, py, true);
        SET_VECTOR_ELT(rlist, 0, rcol1); SET_VECTOR_ELT(rlist, 1, rcol2); SET_VECTOR_ELT(rlist, 2, rval); SET_VECTOR_ELT(rlist, 3, rrank);
        Rf_setAttrib(rlist, R_NamesSymbol, rlist_names); Rf_setAttrib(rlist, R_ClassSymbol, Rf_mkString("data.frame")); Rf_setAttrib(rlist, R_RowNamesSymbol, rnames);
        ans_df = rlist;
    } catch(const std::bad_alloc&) { rerror("Out of memory"); }
      catch(const std::exception &e){ rerror("%s", e.what()); }
    rreturn(ans_df);
}

// ====== CSC-aware KNN (direct-to-sparse + BLAS path for NA-free) ======
SEXP C_bicor_knn_csc(SEXP _x, SEXP _y, SEXP _knn, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, SEXP _direct_sparse, SEXP _envir)
{
    SEXP ans_df = R_NilValue; int __prot_count = 0;
    try {
        TGStat tgstat(_envir);
        const bool haveY = !Rf_isNull(_y);
        if (!(Rf_isS4(_x) && Rf_inherits(_x, "dgCMatrix"))) verror("\"x\" must be a dgCMatrix");
        if (haveY && !(Rf_isS4(_y) && Rf_inherits(_y, "dgCMatrix"))) verror("\"y\" must be a dgCMatrix");
        CSCRef X; load_csc(_x, X); CSCRef Y; if (haveY){ load_csc(_y, Y); if (Y.nrow!=X.nrow) verror("x and y must have same number of rows"); }
        const int n=X.nrow, px=X.ncol, py=haveY?Y.ncol:px;

        const int requested_k = Rf_asInteger(_knn); if (requested_k < 1) verror("knn must be >= 1");
        const int K = std::min(requested_k, std::max(0, px - (haveY ? 0 : 1)));
        const int pairwise_lgl = Rf_asLogical(_pairwise);
        if (pairwise_lgl == NA_LOGICAL) verror("\"pairwise.complete.obs\" must be TRUE or FALSE");
        const bool pairwise_complete = (pairwise_lgl == TRUE);
        const double threshold = Rf_asReal(_threshold);
        const bool   use_inter_den = Rf_asLogical(_use_inter_den);
        const int    min_overlap = getenv_int("BGNS_MIN_OVERLAP", 3);
        bool direct_sparse = Rf_asLogical(_direct_sparse);
        const bool na_free_x = csc_all_finite(X);
        const bool na_free_y = haveY ? csc_all_finite(Y) : na_free_x;
        const bool use_blas = (!use_inter_den) && na_free_x && (!haveY || na_free_y) && n >= min_overlap;

        ProgressReporter progress(TGStat::instance(), (uint64_t)(haveY?py:px));
        vector<int> col1, col2, rankv; vector<double> val;
        const size_t mem_mb = getenv_mb("BGNS_MEM_MB", 256);

        if (use_blas) {
            if (!direct_sparse) {
                const double S = (double)mem_mb * 1024.0 * 1024.0;
                const double need = 8.0*n*px + 8.0*n*py + 8.0*px*py;
                const bool can_full = (need <= S);
                if (can_full) {
                    vector<double> A,B,C,ssX,ssY;
                    { vector<int> cols(px); std::iota(cols.begin(), cols.end(), 0); build_dense_panel_X_csc(X.p,X.i,X.x,n, cols.data(), px, A, ssX); }
                    if (haveY) { vector<int> cols(py); std::iota(cols.begin(), cols.end(), 0); build_dense_panel_X_csc(Y.p,Y.i,Y.x,n, cols.data(), py, B, ssY); }
                    else { B=A; ssY=ssX; }
                    C.assign((size_t)px*(size_t)py, 0.0);
                    dgemm_AB('T','N', px, py, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), px);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((haveY?py:px) >= omp_thresh()) num_threads(bgns_omp_threads())
#endif
                    for (int j=0;j<(haveY?py:px);++j){
                        if (!(ssY[j]>0.0)) { progress.next(); continue; }
                        vector<pair<int,double>> buf; buf.reserve(px);
                        for (int i=0;i<px;++i){
                            if (!haveY && i==j) continue;
                            const double ss_s=ssX[i]; if (!(ss_s>0.0)) continue;
                            const double num=C[(size_t)i + (size_t)j*(size_t)px];
                            const double den=std::sqrt(ss_s * ssY[j]); if (!(den>0.0)) continue;
                            const double r = num / den;
                            tiny_topk_consider(buf, K, i, r, threshold);
                        }
                        if (!buf.empty()){
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
#ifdef _OPENMP
#pragma omp critical
#endif
                            {
                                for (int k=0;k<keep;++k){ col1.push_back(j+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                            }
                        }
                        progress.next();
                    }
                } else { direct_sparse = true; }
            }

            if (direct_sparse) {
                int Pa=1,Pb=1; dense_panel_sizes(n, px, (haveY?py:px), mem_mb, Pa, Pb);
                vector<int> colsA(Pa), colsB(Pb); vector<double> A,B,C,ssA,ssB;
                if (haveY) {
                    for (int j0=0;j0<py;j0+=Pb){ bgns_check_interrupt();
                        const int pb=min(Pb, py-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                        build_dense_panel_X_csc(Y.p,Y.i,Y.x, n, colsB.data(), pb, B, ssB);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            build_dense_panel_X_csc(X.p,X.i,X.x, n, colsA.data(), pa, A, ssA);
                            C.assign((size_t)pa*(size_t)pb, 0.0);
                            dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for (int jb=0;jb<pb;++jb){
                                if (!(ssB[jb]>0.0)) continue;
                                auto &buf=knn[jb];
                                for (int ia=0;ia<pa;++ia){
                                    const int src=i0+ia; const double ss_s=ssA[ia]; if (!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia + (size_t)jb*(size_t)pa];
                                    const double den=std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                    const double r = num / den; tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                if ((int)knn[jb].size()>4*K){ std::nth_element(buf.begin(), buf.begin() + (2*K - 1), buf.end(),
                                    [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); }); buf.resize(2*K); }
                            }
                        }
                        for (int jb=0;jb<pb;++jb){
                            auto &buf=knn[jb]; if (buf.empty()) continue;
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0;k<keep;++k){ col1.push_back(j0+jb+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                        progress.next();
                    }
                } else {
                    for (int j0=0;j0<px;j0+=Pb){ bgns_check_interrupt();
                        const int pb=min(Pb, px-j0); for (int j=0;j<pb;++j) colsB[j]=j0+j;
                        build_dense_panel_X_csc(X.p,X.i,X.x, n, colsB.data(), pb, B, ssB);
                        vector<vector<pair<int,double>>> knn(pb);
                        for (int i0=0;i0<px;i0+=Pa){ bgns_check_interrupt();
                            const int pa=min(Pa, px-i0); for (int i=0;i<pa;++i) colsA[i]=i0+i;
                            build_dense_panel_X_csc(X.p,X.i,X.x, n, colsA.data(), pa, A, ssA);
                            C.assign((size_t)pa*(size_t)pb, 0.0);
                            dgemm_AB('T','N', pa,pb,n, 1.0, A.data(),n, B.data(),n, 0.0, C.data(),pa);
                            for (int jb=0;jb<pb;++jb){
                                const int tgt=j0+jb; if (!(ssB[jb]>0.0)) continue; auto &buf=knn[jb];
                                for (int ia=0;ia<pa;++ia){
                                    const int src=i0+ia; if (src==tgt) continue; const double ss_s=ssA[ia]; if (!(ss_s>0.0)) continue;
                                    const double num=C[(size_t)ia + (size_t)jb*(size_t)pa];
                                    const double den=std::sqrt(ss_s * ssB[jb]); if (!(den>0.0)) continue;
                                    const double r = num / den; tiny_topk_consider(buf, K, src, r, threshold);
                                }
                                if ((int)knn[jb].size()>4*K){ auto &buf=knn[jb]; std::nth_element(buf.begin(), buf.begin() + (2*K - 1), buf.end(),
                                    [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); }); buf.resize(2*K); }
                            }
                        }
                        for (int jb=0;jb<pb;++jb){
                            auto &buf=knn[jb]; if (buf.empty()) continue;
                            const int keep=(int)std::min((size_t)K, buf.size());
                            std::nth_element(buf.begin(), buf.begin()+keep-1, buf.end(),
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            std::sort(buf.begin(), buf.begin()+keep,
                                [](const pair<int,double>& a, const pair<int,double>& b){ return a.second>b.second || (a.second==b.second && a.first<b.first); });
                            for (int k=0;k<keep;++k){ col1.push_back(j0+jb+1); col2.push_back(buf[k].first+1); val.push_back(buf[k].second); rankv.push_back(k+1); }
                        }
                        progress.next();
                    }
                }
            }
        } else {
            // NA-aware CSC path via PanelMixed (treat implicit zeros as finite).
            // This path is shared by bipartite and self-KNN calls so sparse
            // self-KNN remains correct when stored non-finite values are present
            // or when overlap-specific denominators are requested.
            const bool use16 = (n <= 65535);
            const int Pa = use16 ? sparse_panel_cols<uint16_t>(n, px, mem_mb)
                                 : sparse_panel_cols<uint32_t>(n, px, mem_mb);
            const int Pb = use16 ? sparse_panel_cols<uint16_t>(n, py, mem_mb)
                                 : sparse_panel_cols<uint32_t>(n, py, mem_mb);
            vector<int> colsA(Pa), colsB(Pb);
            const CSCRef &T = haveY ? Y : X;

            auto run_na_aware = [&](auto index_tag) {
                using I = decltype(index_tag);
                for (int j0 = 0; j0 < py; j0 += Pb) {
                    bgns_check_interrupt();
                    const int pb = min(Pb, py - j0);
                    for (int j = 0; j < pb; ++j) colsB[j] = j0 + j;

                    PanelMixed<I> TY;
                    build_panel_mixed_from_csc<I>(T.p, T.i, T.x, n,
                                                   colsB.data(), pb, TY);
                    vector<vector<pair<int,double>>> knn(pb);

                    for (int i0 = 0; i0 < px; i0 += Pa) {
                        bgns_check_interrupt();
                        const int pa = min(Pa, px - i0);
                        for (int i = 0; i < pa; ++i) colsA[i] = i0 + i;

                        PanelMixed<I> SX;
                        build_panel_mixed_from_csc<I>(X.p, X.i, X.x, n,
                                                       colsA.data(), pa, SX);

                        for (int tj = 0; tj < TY.P; ++tj) {
                            if (!TY.ok_local[tj]) continue;
                            const int tgt = TY.local_cols[tj];
                            auto &buf = knn[tj];

                            for (int si = 0; si < SX.P; ++si) {
                                if (!SX.ok_local[si]) continue;
                                const int src = SX.local_cols[si];
                                if (!haveY && src == tgt) continue;

                                const double r = panel_mixed_bicor_pair<I>(
                                    SX, si, TY, tj,
                                    pairwise_complete,
                                    use_inter_den,
                                    min_overlap
                                );
                                tiny_topk_consider(buf, K, src, r, threshold);
                            }
                            shrink_topk_buffer(buf, K);
                        }
                    }

                    for (int tj = 0; tj < TY.P; ++tj) {
                        append_final_topk(knn[tj], TY.local_cols[tj], K,
                                          col1, col2, val, rankv);
                    }
                    progress.next();
                }
            };

            if (use16) run_na_aware(uint16_t{});
            else       run_na_aware(uint32_t{});
        }

        const R_xlen_t N = (R_xlen_t)col1.size();
        SEXP rlist, rnames, rlist_names, rcol1, rcol2, rval, rrank;
        rprotect(rlist = RSaneAllocVector(VECSXP, 4));
        rprotect(rlist_names = RSaneAllocVector(STRSXP, 4));
        SET_STRING_ELT(rlist_names, 0, Rf_mkChar("col1"));
        SET_STRING_ELT(rlist_names, 1, Rf_mkChar("col2"));
        SET_STRING_ELT(rlist_names, 2, Rf_mkChar("val"));
        SET_STRING_ELT(rlist_names, 3, Rf_mkChar("rank"));
        rprotect(rcol1 = RSaneAllocVector(INTSXP, N));
        rprotect(rcol2 = RSaneAllocVector(INTSXP, N));
        rprotect(rval  = RSaneAllocVector(REALSXP, N));
        rprotect(rrank = RSaneAllocVector(INTSXP, N));
        rprotect(rnames= RSaneAllocVector(INTSXP, N));
        for (R_xlen_t i=0;i<N;++i){
            INTEGER(rcol1)[i]=col1[i]; INTEGER(rcol2)[i]=col2[i]; REAL(rval)[i]=val[i]; INTEGER(rrank)[i]=rankv[i]; INTEGER(rnames)[i]=(int)(i+1);
        }

        set_knn_factor_levels(rcol1, rcol2, _x, px, _y, py, haveY);

        SET_VECTOR_ELT(rlist, 0, rcol1); SET_VECTOR_ELT(rlist, 1, rcol2); SET_VECTOR_ELT(rlist, 2, rval); SET_VECTOR_ELT(rlist, 3, rrank);
        Rf_setAttrib(rlist, R_NamesSymbol, rlist_names); Rf_setAttrib(rlist, R_ClassSymbol, Rf_mkString("data.frame")); Rf_setAttrib(rlist, R_RowNamesSymbol, rnames);
        ans_df = rlist;
    } catch(const std::bad_alloc&) { rerror("Out of memory"); }
      catch(const std::exception &e){ rerror("%s", e.what()); }
    rreturn(ans_df);
}

// ====== KNN dispatchers ======
SEXP C_bicor_knn(SEXP _x, SEXP _y, SEXP _knn, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, SEXP _envir)
{ return bicor_knn_impl(_x, _y, _knn, _pairwise, _threshold, _use_inter_den, /*direct_sparse=*/true, _envir); }

SEXP C_bicor_knn_opts(SEXP _x, SEXP _y, SEXP _knn, SEXP _pairwise, SEXP _threshold, SEXP _use_inter_den, SEXP _direct_sparse, SEXP _envir)
{
    const bool direct_sparse = Rf_asLogical(_direct_sparse);
    return bicor_knn_impl(_x, _y, _knn, _pairwise, _threshold, _use_inter_den, direct_sparse, _envir);
}

} // extern "C"
