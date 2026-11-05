#include <math.h>
#include "voiage_v1.h"
int main(void) {
  const double x[] = {10, 1, 2, 8}; double evpi = -1, enbs = -1;
  if (voiage_v1_evpi(x, 2, 2, &evpi) != VOIAGE_V1_STATUS_OK) return 1;
  if (voiage_v1_enbs(1.0, 0.5, &enbs) != VOIAGE_V1_STATUS_OK) return 2;
  if (fabs(evpi - 3.0) > 1e-12 || fabs(enbs - 0.5) > 1e-12) return 3;
  return 0;
}
