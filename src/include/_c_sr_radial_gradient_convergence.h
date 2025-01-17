#ifndef _C_SR_RADIAL_GRADIENT_CONVERGENCE_H
#define _C_SR_RADIAL_GRADIENT_CONVERGENCE_H

#include <math.h>

double _c_calculate_dw(double distance, double tSS) {
  return pow((distance * exp((-distance * distance) / tSS)), 4);
}

double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
  float Dk = fabs(Gy * dx - Gx * dy) / sqrt(Gx * Gx + Gy * Gy);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

#endif
