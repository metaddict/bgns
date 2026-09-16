// src/init.c  -- central registration for bgns
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <Rversion.h>

extern SEXP C_bicor(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_bicor_knn(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_bicor_knn_opts(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_bicor_tidy(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_bicor_knn_csc(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP C_bicor_knn_spdem(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"C_bicor",          (DL_FUNC) &C_bicor,          5},
    {"C_bicor_knn",      (DL_FUNC) &C_bicor_knn,      7},
    {"C_bicor_knn_opts", (DL_FUNC) &C_bicor_knn_opts, 8},
    {"C_bicor_tidy",     (DL_FUNC) &C_bicor_tidy,     6},
    {"C_bicor_knn_csc",  (DL_FUNC) &C_bicor_knn_csc,  8},
    {"C_bicor_knn_spdem",(DL_FUNC) &C_bicor_knn_spdem,8},
    {NULL, NULL, 0}
};

void R_init_bgns(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
#if defined(R_VERSION) && R_VERSION >= R_Version(3, 0, 0)
    R_forceSymbols(dll, TRUE);
#endif
}
