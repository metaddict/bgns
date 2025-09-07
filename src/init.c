#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP C_bicor(SEXP, SEXP, SEXP, SEXP, SEXP);
SEXP C_bicor_knn(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"C_bicor",     (DL_FUNC) &C_bicor,     5},
  {"C_bicor_knn", (DL_FUNC) &C_bicor_knn, 6},
  {NULL, NULL, 0}
};

void R_init_bgns(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}