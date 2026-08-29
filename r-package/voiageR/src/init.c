#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

extern int voiage_rust_evpi(const double *values, int rows, int columns,
                            double *out);
extern int voiage_rust_enbs(double evsi_result, double research_cost,
                            double *out);

void voiageR_evpi(double *values, int *rows, int *columns, double *out_value,
                  int *out_status) {
  *out_status = voiage_rust_evpi(values, *rows, *columns, out_value);
}

void voiageR_enbs(double *evsi_result, double *research_cost,
                  double *out_value, int *out_status) {
  *out_status = voiage_rust_enbs(*evsi_result, *research_cost, out_value);
}

static const R_CMethodDef c_methods[] = {
    {"voiageR_evpi", (DL_FUNC)&voiageR_evpi, 5},
    {"voiageR_enbs", (DL_FUNC)&voiageR_enbs, 4},
    {NULL, NULL, 0}};

void attribute_visible R_init_voiageR(DllInfo *dll) {
  R_registerRoutines(dll, c_methods, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
